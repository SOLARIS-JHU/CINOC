"""
Wrapper for Centralized 2D Kuramoto-Sivashinsky Dynamics (Pure JAX)
Enables controlled simulations via a policy using native JAX execution.
""" 
import jax
import tesseracts.ks2d.solver as solver

class PDEDynamics2D:
    def __init__(self, policy_apply_fn):
        """
        Initializes the dynamics wrapper for Centralized 2D KS.
        
        Args:
            policy_apply_fn: The .apply method of your ControlNet (JAX/Flax).
                             Note: This policy must now accept 2D inputs (N, N).
        """
        self.policy_apply_fn = policy_apply_fn

    def unroll_controlled(
        self, 
        u_init, 
        xi_fixed, 
        u_target, 
        params, 
        t_steps,
        # Exposing length, resolution, and physics params
        N_grid=128,
        L=64.0,
        dt=0.05,
        sigma=1.0
    ):
        """
        Performs a FULL controlled KS simulation in ONE call (2D).
        
        Args:
            u_init: Initial state of the field (N_grid, N_grid)
            xi_fixed: Fixed positions of actuators (M, 2)  <-- Changed from 1D
            u_target: Target state (N_grid, N_grid)
            params: Policy parameters (PyTree)
            t_steps: Number of simulation steps (static int)
            N_grid: Spatial resolution (static int)
            L: Domain length (float)
            dt: Time step size (float)
            sigma: Actuator width (float)
            
        Returns:
            trajectory: Tuple (u_traj, xi_traj, u_control_traj, v_dummy_traj)
                        u_traj shape: (t_steps, N_grid, N_grid)
        """
        # Ensure inputs are JAX arrays
        u_init = jax.numpy.array(u_init)
        xi_fixed = jax.numpy.array(xi_fixed)
        u_target = jax.numpy.array(u_target)

        # Call the 2D solver function provided in your first snippet
        return solver.solve_with_policy(
            u_init, 
            xi_fixed, 
            u_target, 
            params, 
            self.policy_apply_fn, 
            t_steps,
            N_grid=N_grid,
            L=L,
            dt=dt,
            sigma=sigma
        )