"""
Core Physics Solver for 2D Kuramoto-Sivashinsky Equation using ETDRK4.
"""
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

jax.config.update("jax_enable_x64", True)

def forcing_fn_2d(xi_fixed, u_intensities, N, L, sigma):
    """Calculates the 2D Gaussian influence of STATIC actuators."""
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, y)
    
    def single_actuator(pos, intensity):
        dx = jnp.abs(X - pos[0])
        dx = jnp.minimum(dx, L - dx)
        dy = jnp.abs(Y - pos[1])
        dy = jnp.minimum(dy, L - dy)
        dist_sq = dx**2 + dy**2
        return intensity * jnp.exp(-dist_sq / (2 * sigma**2))
    
    # Check if we have any actuators to process
    if xi_fixed is None or xi_fixed.shape[0] == 0:
        return jnp.zeros((N, N))
        
    forcings = jax.vmap(single_actuator)(xi_fixed, u_intensities)
    return jnp.sum(forcings, axis=0)

def precompute_etdrk4_coeffs(L_linear, dt):
    """
    Precomputes the stability coefficients for ETDRK4.
    """
    ch = L_linear * dt
    
    # Avoid division by zero at k=0 mode
    tol = 1e-4
    is_small = jnp.abs(ch) < tol
    safe_ch = jnp.where(is_small, 1.0, ch) 
    
    # Taylor expansions for small ch to maintain precision
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
    """
    Calculates the nonlinear term -FFT[ (u^2/2)_x + (u^2/2)_y ] = -FFT[ u*u_x + u*u_y ]?
    Actually for KS: N(u) = - (u^2/2)_x  (1D) -> - (u_x^2 + u_y^2)/2 (2D gradient form often used)
    
    Here using the form: N(u) = - grad( |u|^2 / 2 )? 
    No, standard 2D KS form is often: u_t + lap(u) + lap^2(u) + |grad u|^2 / 2 = 0
    So N(u) = - FFT( |grad u|^2 / 2 )
    """
    # 1. Mask inputs (Standard 2/3 rule)
    u_hat_clean = u_hat * dealias_mask
    
    u_x_hat = 1j * kx * u_hat_clean
    u_y_hat = 1j * ky * u_hat_clean
    
    u_x = jnp.fft.irfftn(u_x_hat, s=(N, N))
    u_y = jnp.fft.irfftn(u_y_hat, s=(N, N))
    
    nonlinear_field = 0.5 * (u_x**2 + u_y**2)
    nl_hat = -jnp.fft.rfftn(nonlinear_field)
    
    # 2. Mask outputs (avoids aliasing)
    nl_hat = nl_hat * dealias_mask
    
    # Force mean to be zero (Mass conservation / no drift)
    nl_hat = nl_hat.at[0, 0].set(0.0) 
    
    return nl_hat

def ks_spectral_step_etdrk4(
    u_hat, 
    u_curr_dummy, 
    xi_fixed, 
    u_control, 
    kx, ky,
    etdrk4_coeffs,
    dealias_mask,
    N=128, 
    L=64.0, 
    dt=0.05, 
    sigma=1.0
):
    """ETDRK4 Time Step Wrapper."""
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

@partial(jax.jit, static_argnames=['policy_apply_fn', 't_steps', 'N_grid'])
def solve_with_policy(
    u_init, 
    xi_fixed, 
    u_target, 
    params, 
    policy_apply_fn, 
    t_steps, 
    N_grid=128, 
    L=64.0, 
    dt=0.01, 
    sigma=1.0
):
    """Full simulation loop controllable by a policy."""
    # Setup spectral frequencies
    dx = L / N_grid
    kx_vec = 2 * jnp.pi * jnp.fft.fftfreq(N_grid, d=dx)
    ky_vec = 2 * jnp.pi * jnp.fft.rfftfreq(N_grid, d=dx)
    
    KX, KY = jnp.meshgrid(kx_vec, ky_vec, indexing='ij')
    q_sq = KX**2 + KY**2
    L_linear = q_sq - q_sq**2
    
    # De-aliasing Mask
    k_max_x = jnp.max(jnp.abs(kx_vec))
    k_max_y = jnp.max(jnp.abs(ky_vec))
    mask_x = jnp.abs(KX) < (2.0/3.0 * k_max_x)
    mask_y = jnp.abs(KY) < (2.0/3.0 * k_max_y)
    dealias_mask = (mask_x & mask_y).astype(jnp.float32)

    # Precompute Coeffs
    etdrk4_coeffs = precompute_etdrk4_coeffs(L_linear, dt)
    
    u_hat_init = jnp.fft.rfftn(u_init)

    def step_fn(carry, _):
        u_hat_curr, u_curr = carry
        u_control = policy_apply_fn(params, u_curr, u_target, xi_fixed)
        
        u_hat_next, u_next = ks_spectral_step_etdrk4(
            u_hat_curr, u_curr, xi_fixed, u_control, 
            KX, KY, etdrk4_coeffs, dealias_mask,
            N=N_grid, L=L, dt=dt, sigma=sigma
        )
        
        v_dummy = jnp.zeros_like(u_control) 
        return (u_hat_next, u_next), (u_next, xi_fixed, u_control, v_dummy)

    _, trajectory = jax.lax.scan(
        step_fn, 
        (u_hat_init, u_init), 
        None, 
        length=t_steps
    )
    
    return trajectory
