import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import sys
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Note: We use data_utils to sample fully developed turbulent fields.
# Ensure that generate_spectral_noise/evolve_to_chaos exist as defined.

@partial(jax.jit, static_argnames=['patch_size', 'n_grid'])
def extract_patches_turb_jit(w_phys, xi_norm, patch_size, n_grid):
    """
    JIT-compiled pure function for 2D local patch extraction.
    Extracts physical vorticity and its gradients.
    Uses 'wrap' padding for the periodic boundary conditions of the fluid.
    """
    grad_x, grad_y = jnp.gradient(w_phys)
    half_patch = patch_size // 2

    # Pad with 'wrap' for PERIODIC boundaries
    pad_width = ((half_patch, half_patch), (half_patch, half_patch))
    padded_w = jnp.pad(w_phys, pad_width, mode='wrap')
    padded_gx = jnp.pad(grad_x, pad_width, mode='wrap')
    padded_gy = jnp.pad(grad_y, pad_width, mode='wrap')

    def get_local_obs(xi):
        # Map normalized coordinates [0, 1] to original grid indices.
        i = jax.lax.stop_gradient((xi[0] * n_grid).astype(int))
        j = jax.lax.stop_gradient((xi[1] * n_grid).astype(int))
        
        # Slice patches directly
        p_w = jax.lax.dynamic_slice(padded_w, (i, j), (patch_size, patch_size))
        p_gx = jax.lax.dynamic_slice(padded_gx, (i, j), (patch_size, patch_size))
        p_gy = jax.lax.dynamic_slice(padded_gy, (i, j), (patch_size, patch_size))
        
        # Flatten and concatenate (Shape: 3 * patch_size^2)
        return jnp.concatenate([p_w.flatten(), p_gx.flatten(), p_gy.flatten()])

    return jax.vmap(get_local_obs)(xi_norm)


class Turb2DMARLEnv(gym.Env):
    def __init__(self, pde_dynamics, initial_conditions, n_agents=64, N_grid=64, L=1.0, dt=0.01, substeps=5, max_steps=150):
        """
        Args:
            pde_dynamics: Instance of PDEDynamics2D.
            initial_conditions: Tensor of pre-generated SPECTRAL fields (w_hat).
            n_agents: Number of control actuators (default 8x8 grid = 64).
            N_grid: Spatial resolution (64x64).
            L: Domain length (1.0).
            dt: Macro control timestep.
            substeps: Micro physics timesteps per macro step.
            max_steps: Total control steps per episode.
        """
        super().__init__()
        self.pde = pde_dynamics
        self.w_hat_pool = initial_conditions 
        self.n_agents = n_agents
        self.N_grid = N_grid
        self.L = L
        self.dt = dt
        self.substeps = substeps
        self.max_steps = max_steps
        
        # System parameters (Injected into observation)
        self.viscosity = 5e-4 # Match DPC config
        self.mu = np.array([L, self.viscosity], dtype=np.float32)
        self.n_mu = len(self.mu)
        
        # 2D Actuator Grid Setup (Fixed Positions)
        grid_dim = int(np.sqrt(n_agents))
        x_lin = np.linspace(0, L, grid_dim, endpoint=False) + (L/grid_dim)/2
        xv, yv = np.meshgrid(x_lin, x_lin)
        self.agent_positions = np.stack([xv.flatten(), yv.flatten()], axis=-1)
        self.xi_norm = jnp.array(self.agent_positions / self.L)
        
        # Observation Config
        self.patch_size = 16 # Matches DPC TurbulenceNet
        self.local_y_dim = 3 * (self.patch_size ** 2) # w, grad_x, grad_y
        self.local_obs_dim = self.local_y_dim + self.n_mu
        
        # Action space: 1D per agent (forcing intensity of fixed Gaussian blobs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_agents, self.local_obs_dim), 
            dtype=np.float32
        )
        
        # State tracking (Keep Spectral for physics, convert to Physical for RL)
        self.current_w_hat = None
        self.timestep = 0
        self.rng = jax.random.PRNGKey(0)

    def _get_local_observations(self, w_hat):
        """Converts spectral state to physical state and extracts localized patches."""
        w_phys = jnp.fft.ifft2(w_hat).real
        
        y_local = np.array(extract_patches_turb_jit(
            w_phys, self.xi_norm, self.patch_size, self.N_grid
        ))
        mu_broadcast = np.tile(self.mu, (self.n_agents, 1))
        return np.concatenate([y_local, mu_broadcast], axis=1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            
        self.timestep = 0
        self.rng, subkey = jax.random.split(self.rng)
        
        # Sample directly from the pre-generated SPECTRAL turbulence pool
        idx = jax.random.randint(subkey, (), 0, self.w_hat_pool.shape[0])
        self.current_w_hat = np.array(self.w_hat_pool[idx])
        
        return self._get_local_observations(self.current_w_hat)

    def step(self, joint_action):
        self.timestep += 1
        
        # The unroll wrapper returns trajectories. Since t_steps=1, we pull index [-1].
        w_phys_traj, u_ctrl_traj = self.pde.unroll_controlled(
            w_hat_init=jnp.array(self.current_w_hat),
            xi_fixed=jnp.array(self.agent_positions),
            params=jnp.array(joint_action),
            t_steps=1, 
            N_grid=self.N_grid,
            L=self.L,
            dt=self.dt,
            substeps=self.substeps,
            viscosity=self.viscosity,
            actuator_grid_shape=(int(np.sqrt(self.n_agents)), int(np.sqrt(self.n_agents))),
        )
        
        # In the wrapper, w_phys_traj contains the *Physical* state at the end of the step.
        # However, to step the environment forward iteratively, we need the *Spectral* state.
        # Since the wrapper currently returns w_phys, we must re-transform it to continue the simulation loop accurately, 
        # OR we can assume the physical trajectory is close enough to approximate the next state.
        # Note: If the PDE wrapper is updated to return the raw `w_hat_next` carry, it would avoid this double-FFT.
        
        new_w_phys = np.array(w_phys_traj[-1])
        self.current_w_hat = np.array(jnp.fft.fft2(new_w_phys)) 
        
        # --- PHYSICS BLOW-UP KILL SWITCH ---
        if np.isnan(new_w_phys).any() or np.isinf(new_w_phys).any():
            print(f" [Env] Turbulence PDE Diverged at step {self.timestep}.")
            dummy_obs = np.zeros((self.n_agents, self.local_obs_dim))
            penalty_rewards = np.full((self.n_agents, 1), -100.0)
            return dummy_obs, penalty_rewards, True, {"global_reward": -100.0}

        obs = self._get_local_observations(self.current_w_hat)
        
        # --- REWARD CALCULATION ---
        # Goal: Drive physical vorticity (Enstrophy) to 0.
        global_enstrophy = np.mean(np.square(new_w_phys))
        
        # Local reward: negative mean squared error of the agent's specific vorticity patch
        y_local_w = obs[:, :self.patch_size**2] 
        local_rewards = -np.mean(np.square(y_local_w), axis=-1, keepdims=True)
        
        # Combined reward formulation (Equal weighting for stabilization)
        rewards = 0.5 * local_rewards + 0.5 * (-global_enstrophy)
        
        info = {"global_reward": -global_enstrophy, "global_state": new_w_phys}
        done = self.timestep >= self.max_steps
        
        return obs, rewards, done, info