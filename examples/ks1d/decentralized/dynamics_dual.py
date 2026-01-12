"""
Wrapper for Centralized 1D Kuramoto-Sivashinsky Dynamics (Pure JAX)
Enables controlled simulations via a policy using native JAX execution.
""" 
import jax
import tesseracts.ks1d.solver as solver

class PDEDynamics:
    def __init__(self, policy_apply_fn):
        """
        Initializes the dynamics wrapper for Centralized 1D KS.
        
        Args:
            policy_apply_fn: The .apply method of your ControlNet (JAX/Flax).
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
        N_grid=256,
        L=64.0,
        dt=0.05,
        sigma=1.0
    ):
        """
        Performs a FULL controlled KS simulation in ONE call.
        
        Args:
            u_init: Initial state of the field (N_grid,)
            xi_fixed: Fixed positions of actuators (M,)
            u_target: Target state (N_grid,)
            params: Policy parameters (PyTree)
            t_steps: Number of simulation steps
            N_grid: Spatial resolution (static)
            L: Domain length
            dt: Time step size
            sigma: Actuator width
            
        Returns:
            trajectory: Tuple (u_traj, xi_traj, u_control_traj, v_dummy_traj)
        """
        # Native JAX execution passing through the configuration to the solver
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