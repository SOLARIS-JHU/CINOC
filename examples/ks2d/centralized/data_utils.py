"""
Data Utilities for 2D Kuramoto-Sivashinsky (KS)
Generates chaotic initial conditions by starting from random noise and 
evolving the system autonomously (no control) for a warm-up period.
"""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt

# Enable 64-bit precision to prevent numerical drift in spectral integration
jax.config.update("jax_enable_x64", True)

# --- 1. Solver Components ---

def forcing_fn_2d(xi_fixed, u_intensities, N, L, sigma):
    """Calculates the 2D Gaussian influence of STATIC actuators."""
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    
    # --- CRITICAL FIX: indexing='ij' ---
    # Matches the spectral solver's grid layout (Row=X, Col=Y)
    X, Y = jnp.meshgrid(x, y, indexing='ij') 
    
    def single_actuator(pos, intensity):
        dx = jnp.abs(X - pos[0])
        dx = jnp.minimum(dx, L - dx)
        dy = jnp.abs(Y - pos[1])
        dy = jnp.minimum(dy, L - dy)
        dist_sq = dx**2 + dy**2
        return intensity * jnp.exp(-dist_sq / (2 * sigma**2))
    
    forcings = jax.vmap(single_actuator)(xi_fixed, u_intensities)
    return jnp.sum(forcings, axis=0)

def precompute_etdrk4_coeffs(L_linear, dt):
    """Precomputes the stability coefficients for ETDRK4."""
    ch = L_linear * dt
    
    # Avoid division by zero at k=0 mode
    tol = 1e-4
    is_small = jnp.abs(ch) < tol
    safe_ch = jnp.where(is_small, 1.0, ch) 
    
    f1 = jnp.where(is_small, 1.0 + ch/2.0 + ch**2/6.0, (jnp.exp(ch) - 1.0) / safe_ch)
    f2 = jnp.where(is_small, 0.5 + ch/6.0 + ch**2/24.0, (jnp.exp(ch) - ch - 1.0) / (safe_ch**2))
    f3 = jnp.where(is_small, 1.0/6.0 + ch/24.0 + ch**2/120.0, (jnp.exp(ch) - 0.5*ch**2 - ch - 1.0) / (safe_ch**3))
    
    E = jnp.exp(ch)
    E2 = jnp.exp(ch / 2.0)
    Q = dt * f1
    P1 = dt * (f1 - 3*f2 + 4*f3)
    P2 = dt * (2*f2 - 4*f3) 
    P3 = dt * (2*f2 - 4*f3)
    P4 = dt * (-f2 + 4*f3)
    
    return E, E2, Q, P1, P2, P3, P4

def get_nonlinear(u_hat, kx, ky, dealias_mask, N):
    # 1. Mask inputs (Standard Orszag 2/3 rule)
    u_hat_clean = u_hat * dealias_mask
    
    u_x_hat = 1j * kx * u_hat_clean
    u_y_hat = 1j * ky * u_hat_clean
    
    u_x = jnp.fft.irfftn(u_x_hat, s=(N, N))
    u_y = jnp.fft.irfftn(u_y_hat, s=(N, N))
    
    nonlinear_field = 0.5 * (u_x**2 + u_y**2)
    nl_hat = -jnp.fft.rfftn(nonlinear_field)
    
    # 2. FIX: Mask outputs to prevent high-freq aliasing garbage
    nl_hat = nl_hat * dealias_mask
    
    # Force mean to be zero to prevent drift
    nl_hat = nl_hat.at[0, 0].set(0.0) 
    return nl_hat

def ks_spectral_step_etdrk4(u_hat, u_curr_dummy, xi_fixed, u_control, kx, ky, etdrk4_coeffs, dealias_mask, N=128, L=64.0, dt=0.05, sigma=1.0):
    E, E2, Q, P1, P2, P3, P4 = etdrk4_coeffs
    
    f_field = forcing_fn_2d(xi_fixed, u_control, N, L, sigma)
    f_hat = jnp.fft.rfftn(f_field)
    
    def NL_fn(uh):
        return get_nonlinear(uh, kx, ky, dealias_mask, N) + f_hat

    Nu_n = NL_fn(u_hat)
    a = E2 * u_hat + Q * Nu_n * 0.5
    Na = NL_fn(a)
    b = E2 * u_hat + Q * Na * 0.5
    Nb = NL_fn(b)
    c = E2 * a + Q * (2.0 * Nb - Nu_n) * 0.5 
    Nc = NL_fn(c)
    
    u_hat_next = (E * u_hat + P1 * Nu_n + P2 * Na + P3 * Nb + P4 * Nc)
    u_next = jnp.fft.irfftn(u_hat_next, s=(N, N))
    
    return u_hat_next, u_next

# --- 2. Data Generation Logic ---

def generate_random_noise_2d(key, N_grid, L, scale=1.0):
    """Generates a smooth random initial condition."""
    x = jnp.linspace(0, L, N_grid, endpoint=False)
    X, Y = jnp.meshgrid(x, x)
    
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Mode 1: L-periodic
    phase1_x = jax.random.uniform(k1, minval=0, maxval=2*jnp.pi)
    phase1_y = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    u = jnp.sin(2*jnp.pi*X/L + phase1_x) * jnp.cos(2*jnp.pi*Y/L + phase1_y)
    
    # Mode 2: L/2-periodic
    phase2 = jax.random.uniform(k3, minval=0, maxval=2*jnp.pi)
    u += 0.5 * jnp.sin(4*jnp.pi*X/L + phase2)
    
    # Small noise
    u += 0.05 * jax.random.normal(k4, shape=(N_grid, N_grid))
    
    # Normalize
    u = (u - jnp.mean(u)) 
    u = u / jnp.std(u) * scale
    
    return u

@partial(jax.jit, static_argnames=("N_grid", "L", "warmup_time", "dt"))
def evolve_to_attractor(key, N_grid, L, warmup_time=200.0, dt=0.005):
    """
    Evolves 2D random noise for 'warmup_time' to reach the chaotic attractor.
    Uses dt=0.005 for high stability.
    """
    # 1. Generate seed noise
    u_init = generate_random_noise_2d(key, N_grid, L)
    u_hat = jnp.fft.rfftn(u_init)
    
    # 2. Setup 2D Spectral Grid
    dx = L / N_grid
    kx_vec = 2 * jnp.pi * jnp.fft.fftfreq(N_grid, d=dx)
    ky_vec = 2 * jnp.pi * jnp.fft.rfftfreq(N_grid, d=dx)
    KX, KY = jnp.meshgrid(kx_vec, ky_vec, indexing='ij')
    
    # Linear Operator: (k^2 - k^4)
    q_sq = KX**2 + KY**2
    L_linear = q_sq - q_sq**2
    
    # 3. Precompute ETDRK4 Coeffs & De-aliasing Mask
    etdrk4_coeffs = precompute_etdrk4_coeffs(L_linear, dt)
    
    k_max_x = jnp.max(jnp.abs(kx_vec))
    k_max_y = jnp.max(jnp.abs(ky_vec))
    mask_x = jnp.abs(KX) < (2.0/3.0 * k_max_x)
    mask_y = jnp.abs(KY) < (2.0/3.0 * k_max_y)
    dealias_mask = (mask_x & mask_y).astype(jnp.float32) # Will be cast to float64 automatically
    
    # 4. Determine steps (Must use static args)
    steps = int(warmup_time / dt)
    
    # Dummy inputs for the solver
    xi_dummy = jnp.zeros((1, 2)) 
    u_control_dummy = jnp.zeros(1)

    def warmup_step(carry, _):
        u_hat_curr, u_curr = carry
        
        u_hat_next, u_next = ks_spectral_step_etdrk4(
            u_hat_curr, u_curr, xi_dummy, u_control_dummy,
            kx=KX, ky=KY, etdrk4_coeffs=etdrk4_coeffs, dealias_mask=dealias_mask,
            N=N_grid, L=L, dt=dt
        )
        return (u_hat_next, u_next), None

    # Run the loop
    (u_hat_final, u_final), _ = jax.lax.scan(
        warmup_step,
        (u_hat, u_init),
        None,
        length=steps
    )
    
    return u_final

def get_batch_initial_conditions(key, batch_size, N_grid, L):
    """Generates a batch of fully developed chaotic states."""
    keys = jax.random.split(key, batch_size)
    
    # Bind static arguments using partial
    evolve_fn = partial(
        evolve_to_attractor, 
        N_grid=N_grid, 
        L=L, 
        warmup_time=200.0, 
        dt=0.005 # Match the dt in signature
    )
    
    batch_u = jax.vmap(evolve_fn)(keys)
    return batch_u

if __name__ == "__main__":
    print("Generating 2D KS Chaotic Attractor samples (x64 precision enabled)...")
    
    L_paper = 32.0  
    N_paper = 64   
    
    key = jax.random.PRNGKey(42)
    
    try:
        u_samples = get_batch_initial_conditions(key, batch_size=3, N_grid=N_paper, L=L_paper)
        
        print(f"Generated batch shape: {u_samples.shape}")
        # Expected range: approx -4 to +4
        print(f"Min/Max values: {u_samples.min():.2f}, {u_samples.max():.2f}")
        
        # --- Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i in range(3):
            ax = axes[i]
            u = u_samples[i]
            
            im = ax.imshow(u, extent=[0, L_paper, 0, L_paper], 
                           origin='lower', cmap='RdBu_r', vmin=-3.0, vmax=3.0)
            ax.set_title(f"Sample {i+1}")
            ax.set_xlabel("x")
            if i == 0: ax.set_ylabel("y")
        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='u(x, y)')
        
        plt.suptitle(f"Fully Developed 2D KS Chaos (L={L_paper})", fontsize=16)
        plt.savefig("ks_2d_initial_conditions.png", bbox_inches='tight')
        print("Saved ks_2d_initial_conditions.png")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")