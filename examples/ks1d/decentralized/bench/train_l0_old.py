import jax
import jax.numpy as jnp
import numpy as np
import time
import optax
import os
from flax import serialization
from flax.training import train_state
import sys
from pathlib import Path 

# Project imports
from env_ks_l0 import AdaptedKuramotoSivashinskyEnv as KSEnv 
from model_l0 import MARLPolynomialActor, CentralizedMARLCritic 

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 8
N_GRID = 128
L_DOMAIN = 22.0
DT = 0.05
BATCH_SIZE = 64
MAX_EPISODES = 500
MAX_STEPS = 200
EVAL_INT = 10
TAU = 0.005
GAMMA = 0.99
REG_COEFF = 0.001  # L0 Sparsity strength

# --- Helper: JAX-Native Polynomial Expansion ---
@jax.jit
def get_poly_features(x):
    # Ensure x is treated as (..., features)
    ones = jnp.ones((*x.shape[:-1], 1))
    # Concatenate: [1, x, x^2]
    return jnp.concatenate([ones, x, jnp.square(x)], axis=-1)

# --- Environment Setup ---
def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = KSEnv(dynamics, n_actuators=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, max_steps=MAX_STEPS)

# Dimensions: Patch (42) -> Poly degree 2 expansion: 1 + 42 + 42 = 85
# Dimensions: Global State (130) -> Poly degree 2 expansion: 1 + 130 + 130 = 261
global_state_dim = 130 
poly_dim = 1 + global_state_dim + global_state_dim

# --- JIT Training Functions ---
@jax.jit
def update_step(actor_state, critic_state, target_actor_params, target_critic_params, 
                batch, key):
    g_state, actions, rewards, ng_state, dones = batch
    
    # Pre-process features (Global state expansion)
    poly_g_state = get_poly_features(g_state)
    poly_ng_state = get_poly_features(ng_state)
    
    # 1. Update Critic
    key, noise_key = jax.random.split(key)
    
    # FIX: Wrap target_actor_params in {'params': ...}
    next_joint_actions = MARLPolynomialActor().apply({'params': target_actor_params}, poly_ng_state)
    
    noise = jnp.clip(jax.random.normal(noise_key, next_joint_actions.shape) * 0.2, -0.5, 0.5)
    next_joint_actions = jnp.clip(next_joint_actions + noise, -1.0, 1.0)
    
    # FIX: Wrap target_critic_params in {'params': ...}
    t_q1, t_q2 = CentralizedMARLCritic().apply({'params': target_critic_params}, ng_state, next_joint_actions)
    target_q = rewards + (1.0 - dones) * GAMMA * jnp.minimum(t_q1, t_q2)
    
    def critic_loss_fn(params):
        # FIX: Wrap params in {'params': ...}
        q1, q2 = CentralizedMARLCritic().apply({'params': params}, g_state, actions)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=c_grads)
    
    # 2. Update Actor with L0 Penalty
    def actor_loss_fn(params):
        # FIX: Wrap params in {'params': ...}
        curr_joint_actions = MARLPolynomialActor().apply({'params': params}, poly_g_state)
        
        # FIX: Wrap critic_state.params in {'params': ...}
        q_val = CentralizedMARLCritic().apply({'params': critic_state.params}, g_state, curr_joint_actions)[0]
        
        # Access sparsity regularization logic from your model
        reg_loss = REG_COEFF * jnp.sum(jax.nn.sigmoid(params['SparsePolynomialLayer_0']['log_alpha']))
        return -jnp.mean(q_val) + reg_loss
    
    a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=a_grads)
    
    # 3. Soft Updates
    new_target_actor = jax.tree_util.tree_map(lambda p, tp: p * TAU + tp * (1 - TAU), actor_state.params, target_actor_params)
    new_target_critic = jax.tree_util.tree_map(lambda p, tp: p * TAU + tp * (1 - TAU), critic_state.params, target_critic_params)
    
    return actor_state, critic_state, new_target_actor, new_target_critic, key

# --- Initialization & Saving Helpers ---
def save_checkpoint(state, path):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state.params))

key = jax.random.PRNGKey(0)
key, a_key, c_key = jax.random.split(key, 3)

actor = MARLPolynomialActor()
critic = CentralizedMARLCritic()
a_params = actor.init(a_key, jnp.ones((1, poly_dim)))['params']
c_params = critic.init(c_key, jnp.ones((1, global_state_dim)), jnp.ones((1, N_AGENTS)))['params']

actor_state = train_state.TrainState.create(apply_fn=actor.apply, params=a_params, tx=optax.adam(1e-4))
critic_state = train_state.TrainState.create(apply_fn=critic.apply, params=c_params, tx=optax.adam(3e-4))

target_a_params, target_c_params = a_params, c_params

# Buffer (Numpy) - Streamlined for Global State only
buffer_g_state = np.zeros((100000, global_state_dim))
buffer_act = np.zeros((100000, N_AGENTS))
buffer_rew = np.zeros((100000, 1))
buffer_done = np.zeros((100000, 1))
buf_ptr, buf_size = 0, 0

# --- Training Loop ---
if not os.path.exists('models'): os.makedirs('models')

for episode in range(MAX_EPISODES):
    global_state, _ = env.reset(seed=int(time.time()) + episode)
    ep_reward = 0
    
    for t in range(MAX_STEPS):
        # Select Action
        key, subkey = jax.random.split(key)

        # Expand global state: (130,) -> (261,)
        poly_state = get_poly_features(jnp.array(global_state))
        
        # Apply actor: (261,) -> (8,)
        joint_action_jax = actor.apply({'params': actor_state.params}, poly_state)
        
        # Add exploration noise
        noise = jax.random.normal(subkey, joint_action_jax.shape) * 0.1
        joint_action = np.array(jnp.clip(joint_action_jax + noise, -1.0, 1.0)).flatten()

        # Step Environment (AdaptedKuramotoSivashinskyEnv natively outputs global state)
        next_global_state, step_reward, done, truncated, info = env.step(joint_action)
        
        # Buffer Storage
        buffer_g_state[buf_ptr] = global_state
        buffer_act[buf_ptr] = joint_action
        buffer_rew[buf_ptr] = step_reward
        buffer_done[buf_ptr] = float(done)
        
        buf_ptr = (buf_ptr + 1) % 100000
        buf_size = min(buf_size + 1, 100000)
        
        # Train
        if buf_size > BATCH_SIZE:
            idx = np.random.randint(0, buf_size, BATCH_SIZE)
            batch = jax.tree_util.tree_map(jnp.array, (
                buffer_g_state[idx], buffer_act[idx], buffer_rew[idx], 
                buffer_g_state[(idx+1)%buf_size], buffer_done[idx]
            ))
            actor_state, critic_state, target_a_params, target_c_params, key = \
                update_step(actor_state, critic_state, target_a_params, target_c_params, batch, key)
        
        global_state = next_global_state
        ep_reward += step_reward
        if done: break
        
    if episode % EVAL_INT == 0:
        print(f"Ep {episode} | Reward: {ep_reward/MAX_STEPS:.4f} | Sparsity: {jnp.sum(jax.nn.sigmoid(actor_state.params['SparsePolynomialLayer_0']['log_alpha'])):.2f}")
        save_checkpoint(actor_state, f"models/actor_poly_ep{episode}.msgpack")
        save_checkpoint(critic_state, f"models/critic_central_ep{episode}.msgpack")

print("Training Complete.")