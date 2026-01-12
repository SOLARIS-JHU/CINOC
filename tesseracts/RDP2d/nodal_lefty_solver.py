"""
Nodal-Lefty Reaction-Diffusion Solver for Pattern Control

Based on: Ouchdiri et al., "An optimal-control framework for reaction diffusion 
systems with application to synthetic developmental biology," arXiv:2509.15889v1, 2025.

This implements the activator-inhibitor mechanism from Sekine et al. (2018)
for Nodal-Lefty signaling in synthetic developmental biology.

Key features:
- Two species: Nodal (activator, yn) and Lefty (inhibitor, yl)
- Competitive inhibition kinetics (Hill function)
- Neumann boundary conditions (zero-flux)
- Control via light-inducible production (optogenetics)
"""

import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class NodalLeftyConfig:
    """Configuration for Nodal-Lefty reaction-diffusion system."""
    
    # Grid parameters
    N: int = 80                    # Grid points per dimension (80x80)
    L: float = 800.0               # Domain size [μm]
    
    # Diffusion coefficients [μm²/min]
    D_n: float = 1.96              # Nodal diffusion
    D_l: float = 56.39             # Lefty diffusion (~29× faster)
    
    # Degradation rates [min⁻¹]
    gamma_n: float = 2.37e-3       # Nodal degradation
    gamma_l: float = 5.65e-3       # Lefty degradation
    
    # Hill function parameters
    n_n: float = 2.63              # Hill coefficient for Nodal
    n_l: float = 1.09              # Hill coefficient for Lefty
    k_n: float = 9.28              # Half-saturation for Nodal [nM]
    k_l: float = 14.96             # Half-saturation for Lefty [nM]
    
    # Production parameters (for pattern generation)
    alpha_n: float = 0.8           # Nodal production rate [nM/min]
    alpha_l: float = 4.0           # Lefty production rate [nM/min]
    
    # Control input effects [nM/min when u=1]
    beta_n: float = 0.8            # Nodal control sensitivity
    beta_l: float = 4.0            # Lefty control sensitivity
    
    # Mobile agent parameters
    n_agents: int = 25             # Number of agents
    sigma: float = 20.0            # Gaussian spread [μm]
    V_max: float = 1.0             # Max velocity [μm/min]
    
    # Time stepping
    dt: float = 0.5                # Time step [hours] = 30 min
    t_final: float = 100.0         # Simulation time [hours] for pattern formation


def hill_function(y_n: jnp.ndarray, y_l: jnp.ndarray, config: NodalLeftyConfig) -> jnp.ndarray:
    """
    Competitive inhibition Hill function (regulatory function H).
    
    H(yn, yl) = yn^nn / (yn^nn + [kn^nn * (1 + (yl/kl)^nl)]^nn)
    
    This represents how Nodal activates its own production while 
    Lefty inhibits it through competitive binding.
    """
    # Avoid numerical issues at zero
    y_n_safe = jnp.maximum(y_n, 1e-10)
    y_l_safe = jnp.maximum(y_l, 1e-10)
    
    # Hill terms
    yn_power = y_n_safe ** config.n_n
    
    # Inhibition term: (1 + (yl/kl)^nl)
    inhibition = 1.0 + (y_l_safe / config.k_l) ** config.n_l
    
    # Denominator: yn^nn + [kn^nn * inhibition]^nn
    # But the formula has nested powers - let's be careful
    # Original: yn^nn / (yn^nn + [kn^nn(1 + (yl/kl)^nl)]^nn)
    # This simplifies to: yn^nn / (yn^nn + kn^nn * inhibition^nn)
    
    kn_power = config.k_n ** config.n_n
    denominator = yn_power + kn_power * (inhibition ** config.n_n)
    
    return yn_power / (denominator + 1e-10)


def build_laplacian_neumann(N: int, L: float) -> jnp.ndarray:
    """
    Build spectral Laplacian operator for Neumann (zero-flux) boundary conditions.
    
    Uses DCT (Discrete Cosine Transform) for Neumann BCs.
    """
    dx = L / N
    
    # Wavenumbers for DCT-II (Neumann BC)
    kx = jnp.pi * jnp.arange(N) / L
    ky = jnp.pi * jnp.arange(N) / L
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    
    # Laplacian multiplier in DCT space: -(kx² + ky²)
    laplacian_mult = -(KX**2 + KY**2)
    
    return laplacian_mult


def apply_laplacian_neumann(field: jnp.ndarray, laplacian_mult: jnp.ndarray) -> jnp.ndarray:
    """Apply Laplacian using DCT for Neumann BCs."""
    # DCT-II transform
    field_dct = jax.scipy.fft.dctn(field, type=2, norm='ortho')
    # Multiply by Laplacian
    result_dct = laplacian_mult * field_dct
    # Inverse DCT-II
    return jax.scipy.fft.idctn(result_dct, type=2, norm='ortho')


def build_imex_operators_neumann(
    N: int, L: float, D_n: float, D_l: float, dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build IMEX Crank-Nicolson operators for Neumann BCs.
    
    Uses DCT for spectral representation with zero-flux boundaries.
    """
    laplacian_mult = build_laplacian_neumann(N, L)
    
    # Convert dt from hours to minutes for consistency with parameters
    dt_min = dt * 60.0
    
    # Nodal diffusion operators
    L_n = D_n * laplacian_mult
    implicit_n = 1.0 / (1.0 - (dt_min / 2.0) * L_n)
    explicit_n = 1.0 + (dt_min / 2.0) * L_n
    
    # Lefty diffusion operators
    L_l = D_l * laplacian_mult
    implicit_l = 1.0 / (1.0 - (dt_min / 2.0) * L_l)
    explicit_l = 1.0 + (dt_min / 2.0) * L_l
    
    return implicit_n, explicit_n, implicit_l, explicit_l, laplacian_mult


def imex_step_nodal_lefty(
    y_n: jnp.ndarray,
    y_l: jnp.ndarray,
    xi: jnp.ndarray,           # Agent positions (n_agents, 2)
    u_n_ctrl: jnp.ndarray,     # Nodal control intensity (n_agents,)
    u_l_ctrl: jnp.ndarray,     # Lefty control intensity (n_agents,)
    vel: jnp.ndarray,          # Agent velocities (n_agents, 2)
    xx: jnp.ndarray,
    yy: jnp.ndarray,
    implicit_n: jnp.ndarray,
    explicit_n: jnp.ndarray,
    implicit_l: jnp.ndarray,
    explicit_l: jnp.ndarray,
    config: NodalLeftyConfig
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single IMEX Crank-Nicolson step for Nodal-Lefty system.
    
    The PDEs are:
        ∂yn/∂t = αn·H(yn,yl) + βn·un - γn·yn + Dn·Δyn
        ∂yl/∂t = αl·H(yn,yl) + βl·ul - γl·yl + Dl·Δyl
    """
    dt_min = config.dt * 60.0
    
    # --- Reaction terms (explicit) ---
    H = hill_function(y_n, y_l, config)
    
    R_n = config.alpha_n * H - config.gamma_n * y_n
    R_l = config.alpha_l * H - config.gamma_l * y_l
    
    # --- Control forcing from mobile agents (explicit) ---
    sigma_sq_2 = 2 * config.sigma**2
    norm_factor = 1.0 / (2 * jnp.pi * config.sigma**2)
    
    def single_agent_kernel(xi_i, u_n, u_l):
        """Gaussian kernel for a single agent."""
        dx = xx - xi_i[0]
        dy = yy - xi_i[1]
        sq_dist = dx**2 + dy**2
        kernel = norm_factor * jnp.exp(-sq_dist / sigma_sq_2)
        return config.beta_n * u_n * kernel, config.beta_l * u_l * kernel
    
    B_n_all, B_l_all = jax.vmap(single_agent_kernel)(xi, u_n_ctrl, u_l_ctrl)
    B_n = jnp.sum(B_n_all, axis=0)
    B_l = jnp.sum(B_l_all, axis=0)
    
    # --- IMEX step using DCT for Neumann BCs ---
    # Transform current state
    yn_dct = jax.scipy.fft.dctn(y_n, type=2, norm='ortho')
    yl_dct = jax.scipy.fft.dctn(y_l, type=2, norm='ortho')
    
    # Transform RHS (reaction + control)
    RHS_n_dct = jax.scipy.fft.dctn(R_n + B_n, type=2, norm='ortho')
    RHS_l_dct = jax.scipy.fft.dctn(R_l + B_l, type=2, norm='ortho')
    
    # Crank-Nicolson update
    yn_dct_new = implicit_n * (explicit_n * yn_dct + dt_min * RHS_n_dct)
    yl_dct_new = implicit_l * (explicit_l * yl_dct + dt_min * RHS_l_dct)
    
    # Transform back
    y_n_new = jax.scipy.fft.idctn(yn_dct_new, type=2, norm='ortho')
    y_l_new = jax.scipy.fft.idctn(yl_dct_new, type=2, norm='ortho')
    
    # Enforce non-negativity (concentrations can't be negative)
    y_n_new = jnp.maximum(y_n_new, 0.0)
    y_l_new = jnp.maximum(y_l_new, 0.0)
    
    # --- Agent position update (with boundary reflection) ---
    xi_new = xi + config.dt * 60.0 * vel  # dt in hours, vel in μm/min
    # Reflect at boundaries (Neumann-like for agents)
    xi_new = jnp.abs(xi_new)  # Reflect at 0
    xi_new = config.L - jnp.abs(xi_new - config.L)  # Reflect at L
    xi_new = jnp.clip(xi_new, 0.0, config.L)
    
    return y_n_new, y_l_new, xi_new


@partial(jax.jit, static_argnames=['t_steps', 'N_grid'])
def solve_nodal_lefty(
    yn_init: jnp.ndarray,
    yl_init: jnp.ndarray,
    xi_init: jnp.ndarray,
    t_steps: int,
    N_grid: int = 80,
    L: float = 800.0,
    dt: float = 0.5,
    D_n: float = 1.96,
    D_l: float = 56.39,
    gamma_n: float = 2.37e-3,
    gamma_l: float = 5.65e-3,
    n_n: float = 2.63,
    n_l: float = 1.09,
    k_n: float = 9.28,
    k_l: float = 14.96,
    alpha_n: float = 0.8,
    alpha_l: float = 4.0,
    sigma: float = 20.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve Nodal-Lefty system (uncontrolled) to steady-state pattern.
    """
    config = NodalLeftyConfig(
        N=N_grid, L=L, dt=dt,
        D_n=D_n, D_l=D_l,
        gamma_n=gamma_n, gamma_l=gamma_l,
        n_n=n_n, n_l=n_l, k_n=k_n, k_l=k_l,
        alpha_n=alpha_n, alpha_l=alpha_l,
        sigma=sigma
    )
    
    # Build grid
    x_grid = jnp.linspace(0, L, N_grid)
    y_grid = jnp.linspace(0, L, N_grid)
    xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Build IMEX operators
    implicit_n, explicit_n, implicit_l, explicit_l, _ = build_imex_operators_neumann(
        N_grid, L, D_n, D_l, dt
    )
    
    n_agents = xi_init.shape[0]
    
    def step_fn(carry, _):
        y_n, y_l, xi = carry
        
        # No control
        u_n_ctrl = jnp.zeros(n_agents)
        u_l_ctrl = jnp.zeros(n_agents)
        vel = jnp.zeros((n_agents, 2))
        
        yn_new, yl_new, xi_new = imex_step_nodal_lefty(
            y_n, y_l, xi, u_n_ctrl, u_l_ctrl, vel,
            xx, yy, implicit_n, explicit_n, implicit_l, explicit_l,
            config
        )
        
        return (yn_new, yl_new, xi_new), (yn_new, yl_new, xi_new)
    
    _, trajectory = jax.lax.scan(
        step_fn,
        (yn_init, yl_init, xi_init),
        None,
        length=t_steps
    )
    
    return trajectory


def generate_initial_condition(
    key: jax.Array,
    config: NodalLeftyConfig,
    noise_level: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate random initial condition around homogeneous steady state.
    
    From the paper, steady-state concentrations are O(10-100) nM.
    At steady state: yn_ss = (αn/γn) * H ≈ 50-100 nM when H is active
    """
    key1, key2 = jax.random.split(key)
    
    # Approximate homogeneous steady state (from paper figures)
    # Nodal values: 20-60 nM for spotted, higher for striped
    # Lefty values: 40-150 nM
    # We need yn ~ kn to activate the Hill function
    
    yn_base = 30.0   # nM - around mid-range of paper values
    yl_base = 60.0   # nM - higher for inhibitor
    
    # Add random perturbations to break symmetry
    y_n_init = yn_base + noise_level * yn_base * jax.random.normal(key1, (config.N, config.N))
    y_l_init = yl_base + noise_level * yl_base * jax.random.normal(key2, (config.N, config.N))
    
    # Enforce non-negativity
    y_n_init = jnp.maximum(y_n_init, 1.0)
    y_l_init = jnp.maximum(y_l_init, 1.0)
    
    return y_n_init, y_l_init


def get_pattern_config(case: str) -> Tuple[float, float, str]:
    """
    Get (alpha_n, alpha_l, description) for different pattern types.
    
    Based on Table 1 from Ouchdiri et al. (2025).
    """
    patterns = {
        'striped': (0.8, 4.0, 'Striped pattern'),           # Initial pattern
        'spotted': (0.5, 4.0, 'Spotted pattern'),           # Case 1
        'modified_spotted': (0.8, 4.5, 'Modified spotted'),  # Case 2
        'modified_striped': (1.0, 4.6, 'Modified striped'),  # Case 3
        'radial': (1.5, 8.0, 'Striped/Radial pattern'),     # Case 4
    }
    return patterns.get(case, patterns['striped'])


# =============================================================================
# Demo/Test
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("Nodal-Lefty Reaction-Diffusion Solver")
    print("="*60)
    
    config = NodalLeftyConfig()
    
    print(f"\nParameters:")
    print(f"  Grid: {config.N}×{config.N}, Domain: {config.L}×{config.L} μm")
    print(f"  Diffusion: Dn={config.D_n}, Dl={config.D_l} μm²/min")
    print(f"  Ratio Dl/Dn = {config.D_l/config.D_n:.1f} (Lefty faster)")
    print(f"  Production: αn={config.alpha_n}, αl={config.alpha_l} nM/min")
    
    # Initial condition
    key = jax.random.PRNGKey(42)
    yn_init, yl_init = generate_initial_condition(key, config)
    
    n_agents = 4
    key, subkey = jax.random.split(key)
    xi_init = jax.random.uniform(subkey, (n_agents, 2)) * config.L
    
    # Evolve to pattern
    t_steps = 200  # 100 hours
    print(f"\nEvolving for {t_steps * config.dt} hours...")
    
    (yn_hist, yl_hist, xi_hist) = solve_nodal_lefty(
        yn_init, yl_init, xi_init,
        t_steps=t_steps,
        N_grid=config.N,
        L=config.L,
        dt=config.dt,
        D_n=config.D_n,
        D_l=config.D_l,
        gamma_n=config.gamma_n,
        gamma_l=config.gamma_l,
        n_n=config.n_n,
        n_l=config.n_l,
        k_n=config.k_n,
        k_l=config.k_l,
        alpha_n=config.alpha_n,
        alpha_l=config.alpha_l,
        sigma=config.sigma
    )
    
    yn_final = yn_hist[-1]
    yl_final = yl_hist[-1]
    
    print(f"Final Nodal range: [{float(yn_final.min()):.2f}, {float(yn_final.max()):.2f}] nM")
    print(f"Final Lefty range: [{float(yl_final.min()):.2f}, {float(yl_final.max()):.2f}] nM")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, t_idx in enumerate([0, t_steps//2, -1]):
        axes[0, i].imshow(yn_hist[t_idx].T, origin='lower', 
                          extent=[0, config.L, 0, config.L], cmap='Reds')
        axes[0, i].set_title(f't = {t_idx * config.dt:.0f} h' if t_idx >= 0 else 'Final')
        if i == 0:
            axes[0, i].set_ylabel('Nodal [nM]')
        
        axes[1, i].imshow(yl_hist[t_idx].T, origin='lower',
                          extent=[0, config.L, 0, config.L], cmap='Blues')
        if i == 0:
            axes[1, i].set_ylabel('Lefty [nM]')
    
    plt.suptitle('Nodal-Lefty Pattern Formation', fontsize=14)
    plt.tight_layout()
    plt.savefig('nodal_lefty_test.png', dpi=150)
    print("\nSaved nodal_lefty_test.png")
