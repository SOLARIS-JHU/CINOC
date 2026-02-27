import jax
import jax.numpy as jnp
import numpy as np
import time
import os
from flax import serialization
import sys
from pathlib import Path 
from sklearn.preprocessing import PolynomialFeatures

# Project imports
from env_ks_l0 import POMDPAdaptedKuramotoSivashinskyEnv as KSEnv 
from model_l0 import JAXMARLTD3 # Import the wrapper we built!

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 8
N_SENSORS = 8
N_GRID = 128
L_DOMAIN = 22.0
DT = 0.05
BATCH_SIZE = 64
MAX_EPISODES = 500
MAX_STEPS = 200
EVAL_INT = 10
GAMMA = 0.99
REG_COEFF = 0.001  # L0 Sparsity strength (\lambda)

# Dimensions
MU_DIM = 2 # L and dt
OBS_DIM = N_SENSORS + MU_DIM # 10

# --- Helper: Sklearn Polynomial Expansion ---
poly_transformer = PolynomialFeatures(degree=2, include_bias=True)
# Fit a dummy array to initialize shapes
_dummy_obs = np.zeros((1, OBS_DIM))
poly_transformer.fit(_dummy_obs)
POLY_DIM = poly_transformer.transform(_dummy_obs).shape[1]

def get_poly_features(x):
    # Sklearn handles the cross-terms easily on the CPU side
    x = np.atleast_2d(x)
    return poly_transformer.transform(x).flatten()

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.obs = np.zeros((max_size, OBS_DIM))
        self.poly_obs = np.zeros((max_size, POLY_DIM))
        self.actions = np.zeros((max_size, N_AGENTS))
        self.next_obs = np.zeros((max_size, OBS_DIM))
        self.next_poly_obs = np.zeros((max_size, POLY_DIM))
        self.rewards = np.zeros((max_size, 1))
        self.dones = np.zeros((max_size, 1))
        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def add(self, obs, poly_obs, action, next_obs, next_poly_obs, reward, done):
        self.obs[self.ptr] = obs
        self.poly_obs[self.ptr] = poly_obs
        self.actions[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.next_poly_obs[self.ptr] = next_poly_obs
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.poly_obs[idx],
            self.actions[idx],
            self.next_obs[idx],
            self.next_poly_obs[idx],
            self.rewards[idx],
            self.dones[idx]
        )

# --- Environment Setup ---
def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = KSEnv(dynamics, n_actuators=N_AGENTS, n_sensors=N_SENSORS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, max_steps=MAX_STEPS)

# --- Initialize Agent Wrapper ---
agent = JAXMARLTD3(
    poly_feat_dim=POLY_DIM, 
    obs_dim=OBS_DIM, 
    joint_act_dim=N_AGENTS,
    max_action=1.0, 
    tau=0.005, 
    lr=3e-4, 
    seed=0
)

buffer = ReplayBuffer()
rng = jax.random.PRNGKey(42)
total_it = 0

# --- Training Loop ---
if not os.path.exists('models'): os.makedirs('models')

for episode in range(MAX_EPISODES):
    # 1. Reset Env: Get POMDP observation
    obs, _ = env.reset(seed=int(time.time()) + episode)
    
    # 2. Expand the local POMDP observation to polynomial features for the Actor
    poly_obs = np.array(get_poly_features(obs))
    
    ep_reward = 0
    
    for t in range(MAX_STEPS):
        total_it += 1
        
        # Select Action
        action = agent.get_action(poly_obs)
        
        # Add exploration noise
        noise = np.random.normal(0, 0.1, size=action.shape)
        action = np.clip(action + noise, -1.0, 1.0)

        # Step Environment
        next_obs, step_reward, done, truncated, info = env.step(action)
        
        # Check divergence
        if info.get("diverged", False):
            done = True

        # Next States
        next_poly_obs = np.array(get_poly_features(next_obs))
        
        # Store in buffer
        buffer.add(obs, poly_obs, action, next_obs, next_poly_obs, step_reward, done)
        
        # Update trackers
        obs = next_obs
        poly_obs = next_poly_obs
        ep_reward += step_reward
        
        # Train
        if buffer.size > BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            
            # 1. Update Critic
            agent.critic_state, c_loss, rng = agent.update_critic(
                agent.actor_state, agent.critic_state, batch, rng, 
                policy_noise=0.2, noise_clip=0.5, discount=GAMMA
            )

            # 2. Delayed Policy Update
            if total_it % 2 == 0:
                agent.actor_state, agent.critic_state, a_loss = agent.update_actor(
                    agent.actor_state, agent.critic_state, batch, lambda_l0=REG_COEFF
                )
                
        if done or truncated: 
            break
            
    if episode % EVAL_INT == 0:
        # Calculate active parameters (True Sparsity count)
        log_alpha = agent.actor_state.params['sparse_layer']['log_alpha']
        gamma, zeta = -0.1, 1.1
        s_stretched = jax.nn.sigmoid(log_alpha) * (zeta - gamma) + gamma
        gate = jnp.clip(s_stretched, 0.0, 1.0)
        active_params = jnp.sum(gate > 0)
        total_params = POLY_DIM * N_AGENTS
        
        print(f"Ep {episode} | Reward: {ep_reward/MAX_STEPS:.4f} | Active Parameters: {active_params} / {total_params}")
        
        # Save models
        with open(f"models/actor_poly_ep{episode}.msgpack", "wb") as f:
            f.write(serialization.to_bytes(agent.actor_state.params))
        with open(f"models/critic_central_ep{episode}.msgpack", "wb") as f:
            f.write(serialization.to_bytes(agent.critic_state.params))

print("Training Complete.")