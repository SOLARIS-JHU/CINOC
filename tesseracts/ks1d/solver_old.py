import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

# --- Configuration ---
# KS requires a larger domain to exhibit chaos (typically L >= 32)
N_grid = 128  
L = 32.0      
dx = L / N_grid
dt = 0.05     
sigma = 1.0   # Actuator width scaled to domain size

# Wave numbers for Spectral Derivatives
k = 2 * jnp.pi * jnp.fft.rfftfreq(N_grid, d=dx)
# Linear operator L = -k^2 + k^4 (Note: Sign depends on equation convention)
# Using convention: u_t = -u_xx - u_xxxx - u*u_x
# Fourier space: u_t = (k^2 - k^4)*u_hat - FFT(u*u_x)
L_linear = k**2 - k**4

def forcing_fn_1d(xi_fixed, u_intensities, N):
    """
    Calculates the 1D Gaussian influence of STATIC actuators.
    xi_fixed: (M,) fixed positions
    u_intensities: (M,) control inputs
    """
    x_coords = jnp.linspace(0, L, N, endpoint=False)
    
    def single_actuator(pos, intensity):
        # Periodic distance for influence
        dist = jnp.abs(x_coords - pos)
        dist = jnp.minimum(dist, L - dist) # Handle periodicity
        return intensity * jnp.exp(-(dist**2) / (2 * sigma**2))
    
    forcings = jax.vmap(single_actuator)(xi_fixed, u_intensities)
    return jnp.sum(forcings, axis=0)

@jit
def ks_spectral_step(u_hat, u_field, xi_fixed, u_control):
    """
    Semi-Implicit Crank-Nicolson Spectral Step.
    Solves: u_t = -u_xx - u_xxxx - u*u_x + F
    """
    # 1. Non-linear term (computed in real space)
    # u * u_x = 0.5 * d/dx (u^2)
    u_sq = u_field ** 2
    u_sq_hat = jnp.fft.rfft(u_sq)
    nonlinear_term_hat = -0.5 * (1j * k) * u_sq_hat
    
    # 2. Forcing term (computed in real space)
    f_field = forcing_fn_1d(xi_fixed, u_control, N_grid)
    f_hat = jnp.fft.rfft(f_field)

    # 3. Time Stepping (Crank-Nicolson for Linear, Adams-Bashforth/Explicit for Non-linear)
    # (u_new - u_old)/dt = L_linear * (u_new + u_old)/2 + NonLinear + Forcing
    # u_new * (1 - dt*L/2) = u_old * (1 + dt*L/2) + dt * (NonLinear + Forcing)
    
    denom = 1.0 - (dt / 2.0) * L_linear
    numer = (1.0 + (dt / 2.0) * L_linear) * u_hat + dt * (nonlinear_term_hat + f_hat)
    
    u_hat_next = numer / denom
    
    # 4. Recover real space for next iteration/logging
    u_next = jnp.fft.irfft(u_hat_next)
    
    return u_hat_next, u_next

@partial(jax.jit, static_argnums=(4, 5))
def solve_with_policy(u_init, xi_fixed, u_target, params, policy_apply_fn, t_steps):
    """
    KS Loop: Policy determines intensity (u_control) only.
    Actuator positions (xi) are fixed.
    """
    # Initial transform
    u_hat_init = jnp.fft.rfft(u_init)

    def step_fn(carry, _):
        u_hat_curr, u_curr = carry
        
        # 1. Policy Inference
        # Adapted: Policy only returns 'u_control' (intensities), no 'v' (velocity)
        # We pass u_curr (state), u_target, and xi_fixed (static locations)
        u_control = policy_apply_fn(params, u_curr, u_target, xi_fixed)
        
        # 2. KS Physics Step
        u_hat_next, u_next = ks_spectral_step(u_hat_curr, u_curr, xi_fixed, u_control)
        
        # Return strict tuple structure: 
        # State Carry: (Spectra, Real)
        # Trajectory Log: (Real State, Fixed Positions, Control, Zero Velocity)
        # We log 'zero velocity' to maintain vague compatibility with visualization tools that might expect 4 outputs
        v_dummy = jnp.zeros_like(u_control) 
        
        return (u_hat_next, u_next), (u_next, xi_fixed, u_control, v_dummy)

    _, trajectory = jax.lax.scan(
        step_fn, 
        (u_hat_init, u_init), 
        None, 
        length=t_steps
    )
    
    return trajectory # (u_traj, xi_traj, control_traj, v_traj)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- 1. Define a Dummy Policy for testing ---
    def dummy_policy_fn(params, u_curr, u_target, xi_fixed):
        # This policy simply outputs zeros (uncontrolled KS)
        n_actuators = xi_fixed.shape[0]
        return jnp.zeros((n_actuators,))

    # --- 2. Test Setup ---
    # Parameters
    n_steps = 1000
    n_actuators = 4
    actuator_positions = jnp.linspace(0, L, n_actuators, endpoint=False)

    # Initial Condition: Sine wave with some noise
    x = jnp.linspace(0, L, N_grid, endpoint=False)
    u0 = jnp.sin(2 * jnp.pi * x / L) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (N_grid,))
    u_target = jnp.zeros_like(u0) # Target is to suppress the chaos (laminar state)

    # --- 3. Run Simulation ---
    print(f"Running simulation for {n_steps} steps...")
    # In a real scenario, 'params' would be your NN weights
    trajectory = solve_with_policy(u0, actuator_positions, u_target, None, dummy_policy_fn, n_steps)
    u_history, xi_history, control_history, _ = trajectory

    # --- 4. Plotting ---
    plt.figure(figsize=(10, 6))
    # 
    im = plt.imshow(u_history, aspect='auto', extent=[0, L, n_steps * dt, 0], cmap='RdBu_r')
    plt.colorbar(im, label='u(x, t)')
    plt.xlabel('Spatial Domain (x)')
    plt.ylabel('Time (t)')
    plt.title('Kuramoto-Sivashinsky Trajectory')

    # Overlay actuator positions as vertical dashed lines
    for pos in actuator_positions:
        plt.axvline(x=pos, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('ks_trajectory.png')
    print("Plot saved as 'ks_trajectory.png'")
    # plt.show()