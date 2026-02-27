import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class POMDPAdaptedKuramotoSivashinskyEnv(gym.Env):
    """
    Partially Observable (POMDP) version of the JAX-based KS control environment.
    """
    def __init__(self, pde_dynamics, n_actuators=8, n_sensors=8, N_grid=128, L=22.0, dt=0.05, max_steps=200):
        super().__init__()
        
        self.pde = pde_dynamics
        self.n_agents = n_actuators
        self.n_sensors = n_sensors
        self.N_grid = N_grid
        self.L = L
        self.dt = dt
        self.max_steps = max_steps
        
        # Action Space: All actuator strengths
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32)
        
        # POMDP Observation Space: Sensor readings + system parameters (mu: L and dt)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_sensors + 2,), 
            dtype=np.float32
        )

        # Actuator positions (equidistant physical locations)
        self.agent_positions = np.linspace(0.0, L, n_actuators, endpoint=False) + (L/n_actuators)/2
        
        # Sensor indices (equidistant array indices for slicing the state)
        # This maps the number of sensors to specific points on the N_grid
        self.sensor_indices = np.linspace(0, N_grid, n_sensors, endpoint=False, dtype=int)
        
        self.target_state = jnp.zeros(N_grid)
        self.mu = np.array([L, dt], dtype=np.float32)
        
        self.current_state = None
        self.timestep = 0
        self.rng = jax.random.PRNGKey(0)
        
        self.u_sol = []
        self.history_a = []

    def _get_obs(self):
        # POMDP Logic: Slice the full state array using the sensor indices
        sensor_readings = self.current_state[self.sensor_indices]
        # Concatenate the partial observations with the mu parameters
        return np.concatenate([sensor_readings, self.mu])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            
        self.timestep = 0
        self.u_sol = []
        self.history_a = []
        
        self.rng, subkey = jax.random.split(self.rng)
        
        # Maintained: Evolve to attractor so we don't start from zero/noise
        from examples.ks1d.decentralized.data_utils import evolve_to_attractor
        self.current_state = np.array(evolve_to_attractor(
            subkey, self.N_grid, self.L, warmup_time=100.0, dt=self.dt
        ))
        
        self.u_sol.append(self.current_state)
        return self._get_obs(), {}

    def step(self, action):
        self.timestep += 1
        self.rng, subkey = jax.random.split(self.rng)
        
        traj = self.pde.unroll_controlled(
            u_init=jnp.array(self.current_state),
            xi_fixed=jnp.array(self.agent_positions),
            u_target=self.target_state,
            params=jnp.array(action), 
            t_steps=1,
            N_grid=self.N_grid,
            L=self.L,
            dt=self.dt,
            key=subkey
        )
        
        new_state = np.array(traj[0][-1])
        
        # Maintained: Kill Switch (Divergence check)
        if np.isnan(new_state).any() or np.isinf(new_state).any():
            return self._get_obs(), -100.0, True, False, {"diverged": True}

        self.current_state = new_state
        self.u_sol.append(new_state)
        self.history_a.append(action)
        
        # Reward Calculation: Global energy minimization
        # Note: The agent is evaluated on the FULL state, even though it only sees part of it.
        reward = -np.mean(np.square(self.current_state))
        
        done = self.timestep >= self.max_steps
        truncated = False
        
        return self._get_obs(), float(reward), done, truncated, {}

    def render(self, mode='human'):
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