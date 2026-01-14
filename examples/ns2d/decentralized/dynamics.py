"""
Dynamics wrapper for Decentralized NS2D Shape Formation Control

Exact replica of centralized dynamics, but used with decentralized policy.
Provides interface for decentralized policy networks with local patches.
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

# PhiFlow imports (optional, for compatibility)
try:
    from phi.jax.flow import *
except ImportError:
    pass


# =============================================================================
# JAX-Compatible NS2D Step Function
# =============================================================================


def create_velocity_field(
    xi: jnp.ndarray,           # Agent positions (n_agents, 2)
    velocities: jnp.ndarray,   # Agent velocity vectors (n_agents, 2) - the "push" direction
    Nx: int,
    Ny: int,
    sigma: float = 0.15        # Wider influence for velocity
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create velocity field from agent "push" controls.
    
    Each agent creates a local velocity field that pushes smoke in the direction
    specified by its velocity control. This is like having local fans/vents.
    
    Returns: (u_field, v_field) - velocity components (Nx, Ny) each
    """
    x = jnp.linspace(0, 1, Nx)
    y = jnp.linspace(0, 1.25, Ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    def single_agent_velocity(pos, vel):
        # Gaussian influence centered at agent position
        dist_sq = (X - pos[0])**2 + (Y - pos[1])**2
        influence = jnp.exp(-dist_sq / (2 * sigma**2))
        
        # Agent's velocity control creates local fluid velocity
        u_local = vel[0] * influence  # x-component
        v_local = vel[1] * influence  # y-component
        return u_local, v_local
    
    # Vectorized over agents
    u_fields, v_fields = jax.vmap(single_agent_velocity)(xi, velocities)
    
    return jnp.sum(u_fields, axis=0), jnp.sum(v_fields, axis=0)


def bilinear_sample(field: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear interpolation of field at continuous coordinates (x, y).
    
    Args:
        field: (Nx, Ny) array
        x: (Nx, Ny) x-coordinates (can be fractional)
        y: (Nx, Ny) y-coordinates (can be fractional)
    
    Returns:
        Interpolated values at (x, y) with shape (Nx, Ny)
    """
    Nx, Ny = field.shape
    
    # Clamp coordinates to valid range (zero-flux boundary)
    x = jnp.clip(x, 0, Nx - 1.001)
    y = jnp.clip(y, 0, Ny - 1.001)
    
    # Get integer and fractional parts
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, Nx - 1)
    y1 = jnp.minimum(y0 + 1, Ny - 1)
    
    # Fractional parts
    fx = x - x0
    fy = y - y0
    
    # Gather the four neighbors
    f00 = field[x0, y0]
    f01 = field[x0, y1]
    f10 = field[x1, y0]
    f11 = field[x1, y1]
    
    # Bilinear interpolation
    result = (f00 * (1 - fx) * (1 - fy) +
              f10 * fx * (1 - fy) +
              f01 * (1 - fx) * fy +
              f11 * fx * fy)
    
    return result


def ns2d_step_jax(
    smoke: jnp.ndarray,           # (Nx, Ny)
    xi: jnp.ndarray,              # Agent positions (n_agents, 2)
    push_velocities: jnp.ndarray, # Agent push directions (n_agents, 2)
    dt: float = 1.0,
    buoyancy: float = 0.0,        # Disabled for pure fan control
    sigma_push: float = 0.2,      # Wide influence
    Nx: int = 64,
    Ny: int = 80,
    advect_strength: float = 0.3,  # How strongly fans push (0-1)
    natural_advection: float = 0.003,  # Background upward drift velocity
    diffusion_coef: float = 0.01      # Diffusion coefficient for spreading
) -> jnp.ndarray:
    """
    NS2D step with semi-Lagrangian advection for stability.
    
    Fan-only mode: agents push existing smoke, no injection.
    
    Uses backtracing with bilinear interpolation for stable advection.
    Boundary conditions: zero-flux (Neumann).
    
    Natural dynamics: Even without control, smoke will:
    - Drift upward naturally (like hot smoke rising)
    - Diffuse/spread over time
    """
    
    # 2. Create velocity field from agent push controls
    u_control, v_control = create_velocity_field(xi, push_velocities, Nx, Ny, sigma_push)
    
    # 3. Add NATURAL background advection (upward drift like buoyancy/convection)
    # This makes uncontrolled evolution interesting - smoke rises naturally
    v_natural = jnp.ones((Nx, Ny)) * natural_advection
    u_natural = jnp.zeros((Nx, Ny))  # No horizontal drift by default
    
    # 4. Add weak buoyancy (density-dependent upward flow)
    v_buoyancy = buoyancy * smoke
    
    # 5. Combine all velocity components
    u_total = u_control + u_natural
    v_total = v_control + v_natural + v_buoyancy
    
    # Apply control with full strength (advect_strength scales control authority)
    u_total = u_control * advect_strength + u_natural
    v_total = v_control * advect_strength + v_natural + v_buoyancy
    
    # 6. Semi-Lagrangian advection (stable for any CFL)
    # Create grid coordinates
    ix = jnp.arange(Nx)
    iy = jnp.arange(Ny)
    X, Y = jnp.meshgrid(ix, iy, indexing='ij')
    X = X.astype(jnp.float32)
    Y = Y.astype(jnp.float32)
    
    # Scale velocity from normalized domain to grid indices
    # Domain is [0,1] x [0,1.25], grid is Nx x Ny
    u_grid = u_total * (Nx - 1)  # Convert to grid units
    v_grid = v_total * (Ny - 1) / 1.25  # Account for aspect ratio
    

    # CFL limiter: clamp max displacement per timestep
    # Relaxed to allow stronger control authority
    max_cfl = 0.5
    displacement_mag = jnp.sqrt(u_grid**2 + v_grid**2)
    scale_factor = jnp.where(displacement_mag > max_cfl, 
                             max_cfl / (displacement_mag + 1e-8), 
                             1.0)
    u_grid = u_grid * scale_factor
    v_grid = v_grid * scale_factor
    
    # Backtrace: where did the smoke at (X, Y) come from?
    X_src = X - dt * u_grid
    Y_src = Y - dt * v_grid
    
    # Sample smoke from source locations (with clamped boundaries)
    smoke_advected = bilinear_sample(smoke, X_src, Y_src)
    
    # 7. Diffusion - spreads smoke over time (Gaussian kernel convolution)
    # Larger diffusion_coef = more spreading
    # Kernel is parameterized by diffusion coefficient
    center_weight = 1.0 - 4 * diffusion_coef
    edge_weight = diffusion_coef
    kernel = jnp.array([[0.0, edge_weight, 0.0],
                        [edge_weight, center_weight, edge_weight],
                        [0.0, edge_weight, 0.0]])
    from jax.scipy.signal import convolve2d
    smoke_new = convolve2d(smoke_advected, kernel, mode='same')
    
    # 8. NO-FLOW BOUNDARIES: Enforce zero-flux (Neumann) with multi-cell buffer
    # Copy edge values to prevent smoke from leaking out
    # Use 2-cell buffer for stability
    smoke_new = smoke_new.at[0, :].set(smoke_new[2, :])
    smoke_new = smoke_new.at[1, :].set(smoke_new[2, :])
    smoke_new = smoke_new.at[-1, :].set(smoke_new[-3, :])
    smoke_new = smoke_new.at[-2, :].set(smoke_new[-3, :])
    smoke_new = smoke_new.at[:, 0].set(smoke_new[:, 2])
    smoke_new = smoke_new.at[:, 1].set(smoke_new[:, 2])
    smoke_new = smoke_new.at[:, -1].set(smoke_new[:, -3])
    smoke_new = smoke_new.at[:, -2].set(smoke_new[:, -3])
    
    # 9. Clip to valid range (smoke density must be non-negative)
    smoke_new = jnp.clip(smoke_new, 0, 5)
    
    return smoke_new


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


def compute_wasserstein_loss(z_curr: jnp.ndarray, z_target: jnp.ndarray, 
                              epsilon: float = 0.01, n_iters: int = 20) -> jnp.ndarray:
    """
    Compute Sinkhorn approximation to Wasserstein distance.
    
    This is differentiable and better for transport problems than MSE.
    Uses entropic regularization for numerical stability.
    """
    Nx, Ny = z_curr.shape
    
    # Create cost matrix (squared Euclidean distance on grid)
    x_coords = jnp.linspace(0, 1, Nx)
    y_coords = jnp.linspace(0, 1.25, Ny)
    X1, Y1 = jnp.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Flatten distributions
    p = z_curr.flatten() + 1e-8  # Add small epsilon for stability
    q = z_target.flatten() + 1e-8
    
    # Normalize to probability distributions
    p = p / jnp.sum(p)
    q = q / jnp.sum(q)
    
    # Cost matrix: |x_i - x_j|^2
    X1_flat = X1.flatten()
    Y1_flat = Y1.flatten()
    
    # Compute pairwise squared distances (this is expensive for large grids)
    # Use downsampled version for efficiency
    n = len(p)
    
    # For large grids, use sliced Wasserstein instead (1D projections)
    # Here we use a simplified centroid-based approximation
    cx_curr = jnp.sum(X1 * z_curr) / (jnp.sum(z_curr) + 1e-8)
    cy_curr = jnp.sum(Y1 * z_curr) / (jnp.sum(z_curr) + 1e-8)
    cx_target = jnp.sum(X1 * z_target) / (jnp.sum(z_target) + 1e-8)
    cy_target = jnp.sum(Y1 * z_target) / (jnp.sum(z_target) + 1e-8)
    
    # Wasserstein-1 approximation: transport distance + shape matching
    centroid_dist = jnp.sqrt((cx_curr - cx_target)**2 + (cy_curr - cy_target)**2)
    
    # Add variance matching (encourages similar spread)
    var_curr = jnp.sum(z_curr * ((X1 - cx_curr)**2 + (Y1 - cy_curr)**2)) / (jnp.sum(z_curr) + 1e-8)
    var_target = jnp.sum(z_target * ((X1 - cx_target)**2 + (Y1 - cy_target)**2)) / (jnp.sum(z_target) + 1e-8)
    var_diff = jnp.abs(jnp.sqrt(var_curr + 1e-8) - jnp.sqrt(var_target + 1e-8))
    
    return centroid_dist + 0.5 * var_diff


# =============================================================================
# Policy-Controlled Rollout (Memory-Efficient)
# =============================================================================

@partial(jax.jit, static_argnames=['policy_apply_fn', 't_steps', 'Nx', 'Ny', 'n_agents'])
def unroll_with_full_loss(
    smoke_init: jnp.ndarray,
    xi_init: jnp.ndarray,
    rho_target: jnp.ndarray,
    params,
    policy_apply_fn: Callable,
    t_steps: int,
    Nx: int = 64,
    Ny: int = 80,
    n_agents: int = 9,
    dt: float = 1.0,
    buoyancy: float = 0.3,
    sigma_push: float = 0.15,
    push_max: float = 0.5,    # Max push velocity
    R_safe: float = 0.12,
    domain_margin: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray, float, float, float, float, float]:
    """
    Controlled rollout with fan-only velocity control.
    
    Policy outputs: push_velocity (n_agents, 2)
    - push_velocity: direction to push smoke AND slow agent drift
    
    Returns:
        smoke_final, xi_final, l_track, l_effort, l_bound, l_coll, l_accel
    """
    
    def step_fn(carry, _):
        smoke, xi = carry
        
        # Policy inference - outputs push velocity
        push_vel = policy_apply_fn(params, smoke, rho_target, xi)
        
        # Clip push velocity
        push_norm = jnp.linalg.norm(push_vel, axis=-1, keepdims=True)
        push_vel = jnp.where(push_norm > push_max, push_vel * push_max / (push_norm + 1e-8), push_vel)
        
        # Physics step with velocity control (no injection)
        smoke_new = ns2d_step_jax(
            smoke, xi, push_vel,
            dt=dt, buoyancy=buoyancy,
            sigma_push=sigma_push, Nx=Nx, Ny=Ny
        )
        
        # Mobile agents: move slowly in push direction
        xi_new = xi + dt * push_vel * 0.01
        xi_new = jnp.clip(xi_new, domain_margin, jnp.array([1.0 - domain_margin, 1.25 - domain_margin]))
        
        return (smoke_new, xi_new), (smoke_new, xi_new, push_vel)
    
    # Run rollout
    init_carry = (smoke_init, xi_init)
    (smoke_final, xi_final), trajectories = jax.lax.scan(
        step_fn, init_carry, None, length=t_steps
    )
    
    smoke_traj, xi_traj, v_traj = trajectories
    
    # =========================================================================
    # NEW OBJECTIVE DESIGN: Transport + Hold + Find + Regularization
    # =========================================================================
    
    # Helper: Compute centroid of a field
    def compute_centroid(field):
        Nx_f, Ny_f = field.shape
        x_coords = jnp.linspace(0, 1, Nx_f)
        y_coords = jnp.linspace(0, 1.25, Ny_f)
        X, Y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        total_mass = jnp.sum(field) + 1e-8
        cx = jnp.sum(field * X) / total_mass
        cy = jnp.sum(field * Y) / total_mass
        return jnp.array([cx, cy])
    
    # Target centroid (fixed)
    target_centroid = compute_centroid(rho_target)
    
    # =========================================================================
    # 1. TRANSPORT LOSS: Push smoke centroid → target centroid
    # =========================================================================
    def transport_loss_at_t(smoke_t):
        smoke_centroid = compute_centroid(smoke_t)
        return jnp.sum((smoke_centroid - target_centroid) ** 2)
    
    transport_losses = jax.vmap(transport_loss_at_t)(smoke_traj)
    l_transport = jnp.mean(transport_losses)
    
    # =========================================================================
    # 2. TIME-WEIGHTED HOLD LOSS using WASSERSTEIN DISTANCE
    # =========================================================================
    def hold_loss_at_t(smoke_t):
        return compute_wasserstein_loss(smoke_t, rho_target)
    
    hold_losses = jax.vmap(hold_loss_at_t)(smoke_traj)
    
    # Time weights: [0.5, 0.6, ..., 1.5, 2.0, 2.5] - increasing over time
    # time_weights = 0.5 + 2.0 * jnp.linspace(0, 1, t_steps)  # Linear increase
    # time_weights = 1.0
    time_weights = jnp.ones(t_steps)
    l_hold = jnp.sum(hold_losses * time_weights) / jnp.sum(time_weights)
    
    # Also add strong terminal penalty (Wasserstein)
    l_hold_terminal = compute_wasserstein_loss(smoke_final, rho_target)
    l_hold = l_hold + 3.0 * l_hold_terminal  # Extra emphasis on final state
    
    # =========================================================================
    # 3. FIND LOSS: Agents should approach smoke to push it
    # =========================================================================
    def find_loss_at_t(smoke_t, xi_t):
        smoke_centroid = compute_centroid(smoke_t)
        dist_to_smoke = jnp.sqrt(jnp.sum((xi_t - smoke_centroid) ** 2, axis=-1) + 1e-8)
        return jnp.mean(jnp.minimum(dist_to_smoke, 0.4))
    
    find_losses = jax.vmap(find_loss_at_t)(smoke_traj, xi_traj)
    l_find = jnp.mean(find_losses)
    
    # =========================================================================
    # 4. SURROUND LOSS: Agents should be near TARGET to form containment ring
    # =========================================================================
    def surround_loss_at_t(xi_t):
        # Distance from each agent to TARGET centroid (not smoke!)
        dist_to_target = jnp.sqrt(jnp.sum((xi_t - target_centroid) ** 2, axis=-1) + 1e-8)
        # Ideal distance: ~0.2-0.3 (ring around target, not on top of it)
        ideal_dist = 0.25
        return jnp.mean(jnp.abs(dist_to_target - ideal_dist))  # Encourage ring formation
    
    surround_losses = jax.vmap(surround_loss_at_t)(xi_traj)
    l_surround = jnp.mean(surround_losses)
    
    # =========================================================================
    # 5. BRAKE LOSS: Slow down when smoke is at target (increased threshold)
    # =========================================================================
    def brake_loss_at_t(smoke_t, v_t):
        smoke_centroid = compute_centroid(smoke_t)
        smoke_target_dist = jnp.sqrt(jnp.sum((smoke_centroid - target_centroid) ** 2) + 1e-8)
        
        # Brake factor: 1.0 when at target, 0.0 when far
        # INCREASED threshold from 0.15 to 0.3
        at_target = jnp.clip(1.0 - smoke_target_dist / 0.3, 0.0, 1.0)
        
        # Penalize velocity when at target
        velocity_mag = jnp.mean(jnp.sum(v_t ** 2, axis=-1))
        return at_target * velocity_mag
    
    brake_losses = jax.vmap(brake_loss_at_t)(smoke_traj, v_traj)
    l_brake = jnp.mean(brake_losses)
    
    # =========================================================================
    # REGULARIZATION LOSSES
    # =========================================================================
    
    # 6. COLLISION: Avoid other agents (SOFT EXPONENTIAL - always has gradient)
    diff = xi_traj[:, :, None, :] - xi_traj[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    mask = jnp.eye(n_agents)[None, :, :]
    dists_masked = dists + mask * 10.0  # Ignore self-distance
    
    # Exponential decay: strong when close, weak when far
    # exp(-dist/scale) where scale controls the range of influence
    scale = R_safe  # Use R_safe as scale parameter
    l_coll = jnp.mean(jnp.exp(-dists_masked / scale))
    
    # 7. BOUNDARY: Stay in domain
    x_penalty = jnp.maximum(0, domain_margin - xi_traj[:, :, 0])**2 + \
                jnp.maximum(0, xi_traj[:, :, 0] - (1.0 - domain_margin))**2
    y_penalty = jnp.maximum(0, domain_margin - xi_traj[:, :, 1])**2 + \
                jnp.maximum(0, xi_traj[:, :, 1] - (1.25 - domain_margin))**2
    l_bound = jnp.mean(x_penalty + y_penalty)
    
    # 8. SMOOTH: Velocity smoothness (acceleration penalty)
    l_smooth = jnp.mean(jnp.sum(jnp.diff(v_traj, axis=0)**2, axis=-1))
    
    # 9. EFFORT: Control energy
    l_effort = jnp.mean(jnp.sum(v_traj ** 2, axis=-1))
    
    # 10. MASS CONSERVATION: Penalize loss of smoke mass
    initial_mass = jnp.sum(smoke_init)
    def mass_loss_at_t(smoke_t):
        current_mass = jnp.sum(smoke_t)
        return ((current_mass - initial_mass) / (initial_mass + 1e-8)) ** 2
    
    mass_losses = jax.vmap(mass_loss_at_t)(smoke_traj)
    l_mass = jnp.mean(mass_losses)
    
    return smoke_final, xi_final, l_transport, l_hold, l_find, l_surround, l_brake, l_coll, l_bound, l_smooth, l_effort, l_mass




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
    buoyancy: float = 0.3,
    sigma_push: float = 0.15,
    push_max: float = 0.5
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full controlled rollout with fan-only velocity control for visualization.
    
    Returns:
        smoke_traj (t_steps, Nx, Ny)
        xi_traj (t_steps, n_agents, 2) - agent positions
        push_vel_traj (t_steps, n_agents, 2) - push velocity controls
    """
    
    def step_fn(carry, _):
        smoke, xi = carry
        
        # Policy inference - outputs push velocity
        push_vel = policy_apply_fn(params, smoke, rho_target, xi)
        
        # Clip push velocity
        push_norm = jnp.linalg.norm(push_vel, axis=-1, keepdims=True)
        push_vel = jnp.where(push_norm > push_max, push_vel * push_max / (push_norm + 1e-8), push_vel)
        
        # Physics step (no injection)
        smoke_new = ns2d_step_jax(
            smoke, xi, push_vel,
            dt=dt, buoyancy=buoyancy,
            sigma_push=sigma_push, Nx=Nx, Ny=Ny
        )
        
        # Mobile agents: move slowly in push direction
        xi_new = xi + dt * push_vel * 0.01
        xi_new = jnp.clip(xi_new, 0.1, jnp.array([0.9, 1.15]))
        
        return (smoke_new, xi_new), (smoke_new, xi_new, push_vel)
    
    _, trajectory = jax.lax.scan(
        step_fn,
        (smoke_init, xi_init),
        None,
        length=t_steps
    )
    
    return trajectory


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Decentralized NS2D dynamics wrapper...")
    
    Nx, Ny = 64, 80
    n_agents = 9
    t_steps = 50
    
    # Dummy policy (zero control - fan only)
    def dummy_policy(params, smoke, target, xi):
        n = xi.shape[0]
        return jnp.zeros((n, 2))  # Just push velocity
    
    # Initial conditions
    smoke_init = jnp.zeros((Nx, Ny))
    n_side = int(np.sqrt(n_agents))
    xi_init = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.2, 0.8, n_side),
        jnp.linspace(0.2, 1.0, n_side)
    ), axis=-1).reshape(-1, 2)
    rho_target = jnp.zeros((Nx, Ny))
    
    # Test rollout using unroll_controlled
    smoke_traj, xi_traj, v_traj = unroll_controlled(
        smoke_init, xi_init, rho_target, None, dummy_policy, t_steps,
        Nx=Nx, Ny=Ny
    )
    
    print(f"Smoke trajectory shape: {smoke_traj.shape}")
    print(f"Agent trajectory shape: {xi_traj.shape}")
    print(f"Push velocity shape: {v_traj.shape}")
    print(f"Smoke range: [{float(smoke_traj.min()):.3f}, {float(smoke_traj.max()):.3f}]")
    print("Done!")
