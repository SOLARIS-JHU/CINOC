import jax
import jax.numpy as jnp
import optax
import flax.serialization
import numpy as np
import time
from pathlib import Path
import jax.tree_util

# Project imports
from env_ks import KSHypeMARLEnv
from models_marl import MARLActor, MARLCritic
from utils_hypemarl import get_sinusoidal_encoding, DecentralizedReplayBuffer
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 8
L_DOMAIN = 22.0
N_GRID = 128
BATCH_SIZE = 32
MAX_EPISODES = 500
WARMUP_EPISODES = 25
MAX_STEPS = 200
EVAL_INT = 10
EVAL_SEED = 1234

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = KSHypeMARLEnv(dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=MAX_STEPS)

# Dimensions
local_y_dim = 40 
n_mu = env.n_mu
pe_dim = 2048
# Concatenated input: [patch (40) + mu (2) + pe (2048)]
total_input_dim = local_y_dim + n_mu + pe_dim

# Models
actor = MARLActor()
critic = MARLCritic()

key, *subkeys = jax.random.split(key, 4)
dummy_input = jnp.zeros((BATCH_SIZE, total_input_dim))
dummy_u = jnp.zeros((BATCH_SIZE, 1))

actor_params = actor.init(subkeys[0], dummy_input)
critic_params = critic.init(subkeys[1], dummy_input, dummy_u)

target_actor_params = actor_params
target_critic_params = critic_params

# Optimizers with Gradient Clipping
tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# Static Positional Encoding
pe = np.array(get_sinusoidal_encoding(jnp.array(env.agent_positions), d=pe_dim))

# Buffer stores the fully concatenated state
buffer = DecentralizedReplayBuffer(max_size=100000, obs_dim=total_input_dim, action_dim=1, n_mu=0)

# --- JIT Training Functions ---

@jax.jit
def train_step(a_params, c_params, t_a_params, t_c_params, opt_a, opt_c, x, u, r, nx, key):
    # 1. Target Q
    key, noise_key = jax.random.split(key)
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * 0.2, -0.5, 0.5)
    next_u = jnp.clip(actor.apply(t_a_params, nx) + noise, -1.0, 1.0)
    
    t_q1, t_q2 = critic.apply(t_c_params, nx, next_u)
    target_q = r + 0.99 * jnp.minimum(t_q1, t_q2)
    
    # 2. Update Critic
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, x, u)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    c_loss, c_grads = jax.value_and_grad(c_loss_fn)(c_params)
    c_up, opt_c = tx_critic.update(c_grads, opt_c)
    c_params = optax.apply_updates(c_params, c_up)
    
    # 3. Update Actor
    def a_loss_fn(p):
        q1_out = critic.apply(c_params, x, actor.apply(p, x))[0]
        return -jnp.mean(q1_out)
    
    a_loss, a_grads = jax.value_and_grad(a_loss_fn)(a_params)
    a_up, opt_a = tx_actor.update(a_grads, opt_a)
    a_params = optax.apply_updates(a_params, a_up)
    
    # 4. Soft Updates
    tau = 0.005
    t_a_params = jax.tree_util.tree_map(lambda x, y: tau*x + (1-tau)*y, a_params, t_a_params)
    t_c_params = jax.tree_util.tree_map(lambda x, y: tau*x + (1-tau)*y, c_params, t_c_params)
    
    return a_params, c_params, t_a_params, t_c_params, opt_a, opt_c

@jax.jit
def get_action(params, x, key=None):
    u = actor.apply(params, x)
    if key is not None:
        u = jnp.clip(u + jax.random.normal(key, u.shape) * 0.1, -1.0, 1.0)
    return u

# --- Main Training Loop ---
print("Starting Standard MARL Training (Shared MLP + PE)")

for episode in range(MAX_EPISODES):
    # Noise-free Eval
    if episode % EVAL_INT == 0:
        obs = env.reset(seed=EVAL_SEED + episode)
        eval_e = 0.0
        for _ in range(MAX_STEPS):
            # Concatenate patch + mu + pe
            x_input = np.concatenate([obs, pe], axis=-1)
            action = np.array(get_action(actor_params, x_input))
            obs, _, done, info = env.step(action.flatten())
            eval_e += -info['global_reward']
            if done: break
        print(f"Eval Ep {episode:03d} | Energy: {eval_e/MAX_STEPS:.6f}")

    # Real Interaction
    obs = env.reset(seed=int(time.time()) + episode)
    for t in range(MAX_STEPS):
        x_input = np.concatenate([obs, pe], axis=-1)
        key, subkey = jax.random.split(key)
        action = np.array(get_action(actor_params, x_input, subkey))
        
        n_obs, rew, done, info = env.step(action.flatten())
        
        # Kill switch safeguard
        if done and info.get('global_reward') == -100.0: break
        
        nx_input = np.concatenate([n_obs, pe], axis=-1)
        buffer.add(x_input, action, rew, nx_input, np.zeros((N_AGENTS, 0)))
        obs = n_obs
        
        if buffer.size > BATCH_SIZE:
            bx, bu, br, bnx, _ = buffer.sample(BATCH_SIZE)
            key, subkey = jax.random.split(key)
            actor_params, critic_params, target_actor_params, target_critic_params, opt_actor, opt_critic = \
                train_step(actor_params, critic_params, target_actor_params, target_critic_params, 
                           opt_actor, opt_critic, bx, bu, br, bnx, subkey)

# --- Save Weights ---
with open('models/marl_standard_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params}))
print("Standard MARL training finished and weights saved.")