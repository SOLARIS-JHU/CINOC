"""
Dynamics wrapper for NS2D Shape Formation Control

Provides interface between policy network and PhiFlow NS2D solver.
Enables controlled smoke simulation with movable injection agents.
"""

import sys
from pathlib import Path
from functools import partial
from typing import Callable, Tuple

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

import jax
import jax.numpy as jnp
import numpy as np

# PhiFlow imports
from phi.jax.flow import *


# =============================================================================
# JAX-Compatible NS2D Step Function
# =============================================================================

def create_inflow_field(
    xi: jnp.ndarray,           # Agent positions (n_agents, 2) in [0,1] normalized coords
    intensities: jnp.ndarray,  # Injection intensities (n_agents,)
    Nx: int,
    Ny: int,
    sigma: float = 0.05
) -> jnp.ndarray:
    """
    Create smoke inflow field from Gaussian agents (JAX-compatible).
    
    Returns: Inflow field (Nx, Ny)
    """
    # Grid coordinates in normalized space [0,1] x [0, Ly/Lx]
    x = jnp.linspace(0, 1, Nx)
    y = jnp.linspace(0, 1.25, Ny)  # Domain aspect ratio
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    def single_agent_kernel(pos, intensity):
        dist_sq = (X - pos[0])**2 + (Y - pos[1])**2
        kernel = jnp.exp(-dist_sq / (2 * sigma**2))
        return intensity * kernel
    
    # Vectorized over agents
    inflows = jax.vmap(single_agent_kernel)(xi, intensities)
    return jnp.sum(inflows, axis=0)


def ns2d_step_jax(
    smoke: jnp.ndarray,        # (Nx, Ny)
    velocity: jnp.ndarray,     # (Nx, Ny) - simplified upward velocity magnitude
    xi: jnp.ndarray,           # Agent positions (n_agents, 2)
    intensities: jnp.ndarray,  # Injection intensities (n_agents,)
    dt: float = 1.0,
    buoyancy: float = 0.5,
    sigma: float = 0.05,
    Nx: int = 64,
    Ny: int = 80
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simplified NS2D step using only smoke and scalar velocity field.
    
    This is a simplified model that captures the key control dynamics:
    - Agents inject smoke at their positions
    - Smoke creates upward velocity (buoyancy)
    - Smoke is advected upward and diffuses
    """
    # 1. Create inflow from agents
    inflow = create_inflow_field(xi, intensities, Nx, Ny, sigma)
    
    # 2. Add inflow
    smoke_new = smoke + dt * inflow
    
    # 3. Buoyancy creates upward velocity proportional to smoke
    velocity_new = velocity + dt * buoyancy * smoke_new
    velocity_new = velocity_new * 0.95  # Damping
    
    # 4. Simplified upward advection (smoke moves up based on local velocity)
    # Use roll with weighted mixing based on velocity
    smoke_up = jnp.roll(smoke_new, shift=-1, axis=1)  # Shift upward
    advection_strength = jnp.clip(velocity_new * dt * 0.5, 0, 0.3)
    smoke_new = (1 - advection_strength) * smoke_new + advection_strength * smoke_up
    
    # 5. Simple diffusion (averaging)
    kernel = jnp.array([[0.05, 0.1, 0.05],
                        [0.1,  0.4, 0.1],
                        [0.05, 0.1, 0.05]])
    from jax.scipy.signal import convolve2d
    smoke_new = convolve2d(smoke_new, kernel, mode='same')
    
    # 6. Decay at top boundary to simulate outflow
    y_idx = jnp.arange(Ny) / Ny
    top_decay = jnp.where(y_idx > 0.85, 0.9, 1.0)[None, :]
    smoke_new = smoke_new * top_decay
    
    # 7. Clip to valid range
    smoke_new = jnp.clip(smoke_new, 0, 5)
    
    return smoke_new, velocity_new


# =============================================================================
# Smooth Transport Loss (MSE + Center-of-Mass Guidance)
# =============================================================================

def compute_smooth_loss(z_curr: jnp.ndarray, z_target: jnp.ndarray) -> jnp.ndarray:
    """
    Smooth transport-like loss using MSE + center-of-mass guidance.
    
    This provides gradients even when shapes don't overlap, which is
    critical for transport tasks where initial and target may be far apart.
    
    Args:
        z_curr: Current state (Nx, Ny)
        z_target: Target state (Nx, Ny)
        
    Returns:
        Scalar loss value
    """
    # MSE component (always has gradients)
    mse = jnp.mean((z_curr - z_target) ** 2)
    
    # Add center-of-mass guidance for transport
    eps = 1e-8
    total_curr = jnp.sum(z_curr) + eps
    total_target = jnp.sum(z_target) + eps
    
    # Compute centers of mass
    Nx, Ny = z_curr.shape
    xx, yy = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), indexing='ij')
    
    cx_curr = jnp.sum(xx * z_curr) / total_curr
    cy_curr = jnp.sum(yy * z_curr) / total_curr
    cx_target = jnp.sum(xx * z_target) / total_target
    cy_target = jnp.sum(yy * z_target) / total_target
    
    # Center of mass distance (normalized by grid size)
    com_dist = ((cx_curr - cx_target) ** 2 + (cy_curr - cy_target) ** 2) / (Nx ** 2)
    
    # Combined loss: MSE for local accuracy + COM for global guidance
    return mse + 0.5 * com_dist


# =============================================================================
# Policy-Controlled Rollout (Memory-Efficient)
# =============================================================================

@partial(jax.jit, static_argnames=['policy_apply_fn', 't_steps', 'Nx', 'Ny'])
def unroll_with_loss(
    smoke_init: jnp.ndarray,
    xi_init: jnp.ndarray,
    rho_target: jnp.ndarray,
    params,
    policy_apply_fn: Callable,
    t_steps: int,
    Nx: int = 64,
    Ny: int = 80,
    dt: float = 1.0,
    buoyancy: float = 0.5,
    sigma: float = 0.05,
    u_max: float = 1.0,
    v_max: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    """
    Memory-efficient controlled rollout with integrated loss.
    
    Args:
        smoke_init: Initial smoke density (Nx, Ny)
        xi_init: Initial agent positions (n_agents, 2) in [0,1] coords
        rho_target: Target smoke density (Nx, Ny)
        params: Policy parameters
        policy_apply_fn: Policy function (params, smoke, target, xi) -> (intensities, velocities)
        t_steps: Number of simulation steps
        
    Returns:
        smoke_final, xi_final, tracking_loss, effort_loss
    """
    # Initialize velocity field (simplified - single field)
    velocity = jnp.zeros((Nx, Ny))
    
    @jax.checkpoint
    def step_fn(carry, _):
        smoke, velocity, xi, track_acc, effort_acc = carry
        
        # Policy inference
        intensities, vel = policy_apply_fn(params, smoke, rho_target, xi)
        
        # Clip controls
        intensities = jnp.clip(intensities, 0.0, u_max)
        vel_norm = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        vel = jnp.where(vel_norm > v_max, vel * v_max / (vel_norm + 1e-8), vel)
        
        # Physics step
        smoke_new, velocity_new = ns2d_step_jax(
            smoke, velocity, xi, intensities,
            dt=dt, buoyancy=buoyancy, sigma=sigma, Nx=Nx, Ny=Ny
        )
        
        # Update agent positions
        xi_new = xi + dt * vel
        xi_new = jnp.clip(xi_new, 0.01, jnp.array([0.99, 1.24]))  # Stay in domain
        
        # Accumulate losses (using smooth transport loss with COM guidance)
        l_track = compute_smooth_loss(smoke_new, rho_target)
        l_effort = jnp.mean(intensities**2) + 0.1 * jnp.mean(vel**2)
        
        return (smoke_new, velocity_new, xi_new, 
                track_acc + l_track, effort_acc + l_effort), None
    
    init_carry = (smoke_init, velocity, xi_init, 0.0, 0.0)
    (smoke_final, _, xi_final, total_track, total_effort), _ = jax.lax.scan(
        step_fn, init_carry, None, length=t_steps
    )
    
    return smoke_final, xi_final, total_track / t_steps, total_effort / t_steps


@partial(jax.jit, static_argnames=['policy_apply_fn', 't_steps', 'Nx', 'Ny'])
def unroll_controlled(
    smoke_init: jnp.ndarray,
    xi_init: jnp.ndarray,
    rho_target: jnp.ndarray,
    params,
    policy_apply_fn: Callable,
    t_steps: int,
    Nx: int = 64,
    Ny: int = 80,
    dt: float = 1.0,
    buoyancy: float = 0.5,
    sigma: float = 0.05,
    u_max: float = 1.0,
    v_max: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full controlled rollout returning trajectories for visualization.
    
    Returns:
        smoke_traj (t_steps, Nx, Ny)
        xi_traj (t_steps, n_agents, 2)
        intensity_traj (t_steps, n_agents)
        vel_traj (t_steps, n_agents, 2)
    """
    velocity = jnp.zeros((Nx, Ny))
    
    def step_fn(carry, _):
        smoke, velocity, xi = carry
        
        # Policy inference
        intensities, vel = policy_apply_fn(params, smoke, rho_target, xi)
        
        # Clip controls
        intensities = jnp.clip(intensities, 0.0, u_max)
        vel_norm = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        vel = jnp.where(vel_norm > v_max, vel * v_max / (vel_norm + 1e-8), vel)
        
        # Physics step
        smoke_new, velocity_new = ns2d_step_jax(
            smoke, velocity, xi, intensities,
            dt=dt, buoyancy=buoyancy, sigma=sigma, Nx=Nx, Ny=Ny
        )
        
        # Update agent positions
        xi_new = xi + dt * vel
        xi_new = jnp.clip(xi_new, 0.01, jnp.array([0.99, 1.24]))
        
        return (smoke_new, velocity_new, xi_new), (smoke_new, xi_new, intensities, vel)
    
    _, trajectory = jax.lax.scan(
        step_fn,
        (smoke_init, velocity, xi_init),
        None,
        length=t_steps
    )
    
    return trajectory


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing NS2D dynamics wrapper...")
    
    Nx, Ny = 64, 80
    n_agents = 4
    t_steps = 50
    
    # Dummy policy (zero control)
    def dummy_policy(params, smoke, target, xi):
        n_agents = xi.shape[0]
        return jnp.ones(n_agents) * 0.5, jnp.zeros((n_agents, 2))
    
    # Initial conditions
    key = jax.random.PRNGKey(42)
    smoke_init = jnp.zeros((Nx, Ny))
    xi_init = jnp.array([[0.25, 0.1], [0.4, 0.1], [0.6, 0.1], [0.75, 0.1]])
    rho_target = jnp.zeros((Nx, Ny))
    
    # Test rollout
    smoke_final, xi_final, l_track, l_effort = unroll_with_loss(
        smoke_init, xi_init, rho_target, None, dummy_policy, t_steps,
        Nx=Nx, Ny=Ny
    )
    
    print(f"Smoke final range: [{float(smoke_final.min()):.3f}, {float(smoke_final.max()):.3f}]")
    print(f"Tracking loss: {float(l_track):.4f}")
    print(f"Effort loss: {float(l_effort):.4f}")
    print("Done!")
