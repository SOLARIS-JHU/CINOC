"""
Dynamics wrapper for Nodal-Lefty Reaction-Diffusion System

Provides a clean interface between the policy network and the 
Nodal-Lefty PDE solver for controlled pattern morphing.

"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple
import sys
from pathlib import Path

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from tesseracts.RDP2d.nodal_lefty_solver import (
    NodalLeftyConfig,
    build_imex_operators_neumann,
    imex_step_nodal_lefty,
)


class NodalLeftyDynamics:
    """
    Dynamics wrapper for Nodal-Lefty pattern control.
    """
    
    def __init__(self, policy_apply_fn: Callable, config: NodalLeftyConfig = None):
        """
        Initialize dynamics wrapper.
        
        Args:
            policy_apply_fn: Policy .apply method
                Signature: (params, yn, yl, yn_target, yl_target, xi) -> (un_ctrl, ul_ctrl, vel)
            config: NodalLeftyConfig with system parameters
        """
        self.policy_apply_fn = policy_apply_fn
        self.config = config if config is not None else NodalLeftyConfig()
    
    def unroll_controlled(
        self,
        yn_init: jnp.ndarray,
        yl_init: jnp.ndarray,
        xi_init: jnp.ndarray,
        yn_target: jnp.ndarray,
        yl_target: jnp.ndarray,
        params,
        t_steps: int,
        u_max: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
               jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Run controlled simulation.
        
        Returns:
            yn_traj, yl_traj, xi_traj, un_ctrl_traj, ul_ctrl_traj, vel_traj
        """
        config = self.config
        
        # Build grid
        x_grid = jnp.linspace(0, config.L, config.N)
        y_grid = jnp.linspace(0, config.L, config.N)
        xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Build IMEX operators
        implicit_n, explicit_n, implicit_l, explicit_l, _ = build_imex_operators_neumann(
            config.N, config.L, config.D_n, config.D_l, config.dt
        )
        
        def step_fn(carry, _):
            yn, yl, xi = carry
            
            # Policy inference
            un_ctrl, ul_ctrl, vel = self.policy_apply_fn(
                params, yn, yl, yn_target, yl_target, xi
            )
            
            # Clip controls to [0, 1] (production only, non-negative)
            un_ctrl = jnp.clip(un_ctrl, 0.0, u_max)
            ul_ctrl = jnp.clip(ul_ctrl, 0.0, u_max)
            vel_norm = jnp.linalg.norm(vel, axis=-1, keepdims=True)
            vel = jnp.where(vel_norm > config.V_max, 
                           vel * config.V_max / (vel_norm + 1e-8), vel)
            
            # IMEX step
            yn_new, yl_new, xi_new = imex_step_nodal_lefty(
                yn, yl, xi, un_ctrl, ul_ctrl, vel,
                xx, yy, implicit_n, explicit_n, implicit_l, explicit_l,
                config
            )
            
            return (yn_new, yl_new, xi_new), (yn_new, yl_new, xi_new, un_ctrl, ul_ctrl, vel)
        
        _, trajectory = jax.lax.scan(
            step_fn,
            (yn_init, yl_init, xi_init),
            None,
            length=t_steps
        )
        
        yn_traj, yl_traj, xi_traj, un_ctrl_traj, ul_ctrl_traj, vel_traj = trajectory
        return yn_traj, yl_traj, xi_traj, un_ctrl_traj, ul_ctrl_traj, vel_traj


# =============================================================================
# JIT-compiled version for training/evaluation
# =============================================================================

@partial(jax.jit, static_argnames=['policy_apply_fn', 't_steps', 'N_grid'])
def unroll_jit(
    yn_init: jnp.ndarray,
    yl_init: jnp.ndarray,
    xi_init: jnp.ndarray,
    yn_target: jnp.ndarray,
    yl_target: jnp.ndarray,
    params,
    policy_apply_fn: Callable,
    t_steps: int,
    # Physics parameters
    N_grid: int = 80,
    L: float = 800.0,
    dt: float = 0.5,           # hours
    D_n: float = 1.96,         # μm²/min
    D_l: float = 56.39,        # μm²/min
    gamma_n: float = 2.37e-3,  # min⁻¹
    gamma_l: float = 5.65e-3,  # min⁻¹
    n_n: float = 2.63,
    n_l: float = 1.09,
    k_n: float = 9.28,         # nM
    k_l: float = 14.96,        # nM
    alpha_n: float = 0.5,      # Target alpha (spotted)
    alpha_l: float = 4.0,
    beta_n: float = 0.8,       # Control sensitivity
    beta_l: float = 4.0,
    sigma: float = 40.0,       # μm
    u_max: float = 1.0,
    V_max: float = 1.0         # μm/min
):
    """
    JIT-compiled controlled rollout for Nodal-Lefty evaluation.
    
    No normalization - values are naturally O(10-100) nM.
    Returns trajectory tuple: (yn, yl, xi, un_ctrl, ul_ctrl, vel)
    """
    # Create config
    config = NodalLeftyConfig(
        N=N_grid, L=L, dt=dt,
        D_n=D_n, D_l=D_l,
        gamma_n=gamma_n, gamma_l=gamma_l,
        n_n=n_n, n_l=n_l, k_n=k_n, k_l=k_l,
        alpha_n=alpha_n, alpha_l=alpha_l,
        beta_n=beta_n, beta_l=beta_l,
        sigma=sigma, V_max=V_max
    )
    
    # Build grid
    x_grid = jnp.linspace(0, L, N_grid)
    y_grid = jnp.linspace(0, L, N_grid)
    xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Build IMEX operators
    implicit_n, explicit_n, implicit_l, explicit_l, _ = build_imex_operators_neumann(
        N_grid, L, D_n, D_l, dt
    )
    
    # Normalize inputs for neural network stability
    scale_n = 50.0
    scale_l = 100.0
    scale_L = L
    
    def step_fn(carry, _):
        yn, yl, xi = carry
        
        # Policy inference with normalized values
        un_ctrl, ul_ctrl, vel = policy_apply_fn(
            params, 
            yn / scale_n, 
            yl / scale_l, 
            yn_target / scale_n, 
            yl_target / scale_l, 
            xi / scale_L
        )
        
        # Clip controls to [0, u_max]
        un_ctrl = jnp.clip(un_ctrl, 0.0, u_max)
        ul_ctrl = jnp.clip(ul_ctrl, 0.0, u_max)
        vel_norm = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        vel = jnp.where(vel_norm > V_max, vel * V_max / (vel_norm + 1e-8), vel)
        
        # Velocity is already in physical units from policy (vel_max * tanh)
        vel_physical = vel
        
        # IMEX step
        yn_new, yl_new, xi_new = imex_step_nodal_lefty(
            yn, yl, xi, un_ctrl, ul_ctrl, vel_physical,
            xx, yy, implicit_n, explicit_n, implicit_l, explicit_l,
            config
        )
        
        return (yn_new, yl_new, xi_new), (yn_new, yl_new, xi_new, un_ctrl, ul_ctrl, vel)
    
    _, trajectory = jax.lax.scan(
        step_fn,
        (yn_init, yl_init, xi_init),
        None,
        length=t_steps
    )
    
    return trajectory


# =============================================================================
# Loss-integrated unroll (for memory-efficient BPTT)
# =============================================================================

@partial(jax.jit, static_argnames=['policy_apply_fn', 't_steps', 'N_grid'])
def unroll_with_loss(
    yn_init: jnp.ndarray,
    yl_init: jnp.ndarray,
    xi_init: jnp.ndarray,
    yn_target: jnp.ndarray,
    yl_target: jnp.ndarray,
    params,
    policy_apply_fn: Callable,
    t_steps: int,
    # Physics parameters
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
    alpha_n: float = 0.5,
    alpha_l: float = 4.0,
    beta_n: float = 0.8,
    beta_l: float = 4.0,
    sigma: float = 40.0,
    u_max: float = 1.0,
    V_max: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float]:
    """
    Memory-efficient BPTT: accumulates loss during forward pass.
    
    No normalization - values are naturally O(10-100) nM which works fine.
    
    Returns:
        yn_final, yl_final, xi_final, total_tracking_loss, total_effort_loss
    """
    config = NodalLeftyConfig(
        N=N_grid, L=L, dt=dt,
        D_n=D_n, D_l=D_l,
        gamma_n=gamma_n, gamma_l=gamma_l,
        n_n=n_n, n_l=n_l, k_n=k_n, k_l=k_l,
        alpha_n=alpha_n, alpha_l=alpha_l,
        beta_n=beta_n, beta_l=beta_l,
        sigma=sigma, V_max=V_max
    )
    
    # Build grid
    x_grid = jnp.linspace(0, L, N_grid)
    y_grid = jnp.linspace(0, L, N_grid)
    xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Build operators
    implicit_n, explicit_n, implicit_l, explicit_l, _ = build_imex_operators_neumann(
        N_grid, L, D_n, D_l, dt
    )
    
    # Normalize inputs for neural network stability (O(1) range)
    # This does NOT affect the physics, only the controller's perception
    scale_n = 50.0
    scale_l = 100.0
    scale_L = L
    
    @jax.checkpoint
    def step_with_loss(carry, _):
        yn, yl, xi, track_acc, effort_acc = carry
        
        # Policy sees normalized values for numerical stability
        un_ctrl, ul_ctrl, vel = policy_apply_fn(
            params, 
            yn / scale_n, 
            yl / scale_l, 
            yn_target / scale_n, 
            yl_target / scale_l, 
            xi / scale_L
        )
        
        # Clip controls to [0, 1]
        un_ctrl = jnp.clip(un_ctrl, 0.0, u_max)
        ul_ctrl = jnp.clip(ul_ctrl, 0.0, u_max)
        vel_norm = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        vel = jnp.where(vel_norm > V_max, vel * V_max / (vel_norm + 1e-8), vel)
        
        # Velocity is already in physical units from policy (vel_max * tanh)
        vel_physical = vel
        
        # IMEX step (Physics uses RAW values)
        yn_new, yl_new, xi_new = imex_step_nodal_lefty(
            yn, yl, xi, un_ctrl, ul_ctrl, vel_physical,
            xx, yy, implicit_n, explicit_n, implicit_l, explicit_l,
            config
        )
        
        # Accumulate losses (normalized to O(1) for stable gradients)
        l_track = jnp.mean(((yn_new - yn_target) / scale_n)**2) + jnp.mean(((yl_new - yl_target) / scale_l)**2)
        l_effort = jnp.mean(un_ctrl**2) + jnp.mean(ul_ctrl**2)
        
        return (yn_new, yl_new, xi_new, track_acc + l_track, effort_acc + l_effort), None
    
    init_carry = (yn_init, yl_init, xi_init, 0.0, 0.0)
    (yn_final, yl_final, xi_final, total_track, total_effort), _ = jax.lax.scan(
        step_with_loss, init_carry, None, length=t_steps
    )
    
    return yn_final, yl_final, xi_final, total_track / t_steps, total_effort / t_steps

