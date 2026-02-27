import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

class AdaptedKuramotoSivashinskyEnv(gym.Env):
    """
    Centralized version of KS control, using the physics and parameters 
    from the KSHypeMARLEnv (JAX-based dynamics).
    """
    def __init__(self, pde_dynamics, n_actuators=8, N_grid=128, L=22.0, dt=0.05, max_steps=200):
        super().__init__()
        
        self.pde = pde_dynamics
        self.n_agents = n_actuators
        self.N_grid = N_grid
        self.L = L
        self.dt = dt
        self.max_steps = max_steps
        
        # Centralized Action Space: All actuator strengths
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32)
        
        # Centralized Observation: Full state + system parameters (L and dt)
        # Following the logic of the second env: full state (N_grid) + mu (2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(N_grid + 2,), 
            dtype=np.float32
        )

        # Actuator positions (equidistant)
        self.agent_positions = np.linspace(0.0, L, n_actuators, endpoint=False) + (L/n_actuators)/2
        self.target_state = jnp.zeros(N_grid)
        self.mu = np.array([L, dt], dtype=np.float32)
        
        self.current_state = None
        self.timestep = 0
        self.rng = jax.random.PRNGKey(0)
        
        # Storage for rendering (matching your original env style)
        self.u_sol = []
        self.history_a = []

    def _get_obs(self):
        # Concatenate the 1D field with the parameters
        return np.concatenate([self.current_state, self.mu])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            
        self.timestep = 0
        self.u_sol = []
        self.history_a = []
        
        self.rng, subkey = jax.random.split(self.rng)
        
        # Physics Alignment: Evolve to attractor so we don't start from zero/noise
        from examples.ks1d.decentralized.data_utils import evolve_to_attractor
        self.current_state = np.array(evolve_to_attractor(
            subkey, self.N_grid, self.L, warmup_time=100.0, dt=self.dt
        ))
        
        self.u_sol.append(self.current_state)
        return self._get_obs(), {}

    def step(self, action):
        self.timestep += 1
        self.rng, subkey = jax.random.split(self.rng)
        
        # Physics Call: Use the pde_dynamics class instead of ksintegrate_step
        traj = self.pde.unroll_controlled(
            u_init=jnp.array(self.current_state),
            xi_fixed=jnp.array(self.agent_positions),
            u_target=self.target_state,
            params=jnp.array(action), # Centralized joint action
            t_steps=1,
            N_grid=self.N_grid,
            L=self.L,
            dt=self.dt,
            key=subkey
        )
        
        new_state = np.array(traj[0][-1])
        
        # Kill Switch (Divergence check)
        if np.isnan(new_state).any() or np.isinf(new_state).any():
            return self._get_obs(), -100.0, True, False, {"diverged": True}

        self.current_state = new_state
        self.u_sol.append(new_state)
        self.history_a.append(action)
        
        # Reward Calculation: Global energy minimization
        # Akin to c1 in your original env (negative mean squared state)
        reward = -np.mean(np.square(self.current_state))
        
        done = self.timestep >= self.max_steps
        truncated = False
        
        return self._get_obs(), float(reward), done, truncated, {}

    def render(self, mode='human'):
        # Re-using your original contour plot logic
        fig, ax = plt.subplots(figsize=(10, 4))
        u_plot = np.array(self.u_sol)
        t_axis = np.linspace(0, self.timestep * self.dt, u_plot.shape[0])
        x_axis = np.linspace(0, self.L, self.N_grid)
        
        tt, xx = np.meshgrid(t_axis, x_axis)
        cs = ax.contourf(tt, xx, u_plot.T, cmap=cm.bwr, levels=100)
        fig.colorbar(cs)
        ax.set_xlabel("Time")
        ax.set_ylabel("x")
        plt.show()