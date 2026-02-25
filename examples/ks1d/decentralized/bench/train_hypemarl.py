import jax
import jax.numpy as jnp
import optax
import flax.serialization
import numpy as np
import time
from pathlib import Path
import sys
import jax.tree_util

# Project imports
from examples.ks1d.decentralized.bench.env_ks import KSHypeMARLEnv
from models_hypemarl import HyperActor, HyperCritic, SurrogateModel
from utils_hypemarl import get_sinusoidal_encoding, DecentralizedReplayBuffer

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Centralized JAX wrapper
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
USE_MB_HYPEMARL = False  
N_AGENTS = 8
L_DOMAIN = 22.0
N_GRID = 128
BATCH_SIZE = 32
MAX_EPISODES = 500
WARMUP_EPISODES = 25
MAX_STEPS = 200

# MB-HypeMARL specific
IMAGINARY_RATIO = 10  # Imaginary updates per real episode
IMAGINATION_HORIZON = 50  # Steps to unroll the surrogate model

# Evaluation specific
EVAL_INT = 10
EVAL_SEED = 1234

# --- Initialization ---
key = jax.random.PRNGKey(42)

# 1. Setup Direct Control PDE Dynamics
def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = KSHypeMARLEnv(dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=MAX_STEPS)

# Environment dimensions
local_y_dim = 40  # 20 for error + 20 for grad
n_mu = env.n_mu
pe_dim = 2048
z_dim = pe_dim + n_mu

# 2. Initialize Models
actor = HyperActor()
critic = HyperCritic()
surrogate = SurrogateModel()

key, *subkeys = jax.random.split(key, 6)
dummy_z = jnp.zeros((BATCH_SIZE, z_dim))
dummy_y = jnp.zeros((BATCH_SIZE, local_y_dim))
dummy_u = jnp.zeros((BATCH_SIZE, 1))
dummy_mu = jnp.zeros((BATCH_SIZE, n_mu))

actor_params = actor.init(subkeys[0], dummy_z, dummy_y)
critic1_params = critic.init(subkeys[1], dummy_z, dummy_y, dummy_u)
critic2_params = critic.init(subkeys[2], dummy_z, dummy_y, dummy_u)
surrogate_params = surrogate.init(subkeys[3], dummy_y, dummy_u, dummy_mu)

# Target networks
target_actor_params = actor_params
target_critic1_params = critic1_params
target_critic2_params = critic2_params

# Optimizers 
tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
tx_surrogate = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))

opt_actor = tx_actor.init(actor_params)
opt_critic1 = tx_critic.init(critic1_params)
opt_critic2 = tx_critic.init(critic2_params)
opt_surrogate = tx_surrogate.init(surrogate_params)

# 3. Positional Encoding
pe = get_sinusoidal_encoding(jnp.array(env.agent_positions), d=pe_dim)
pe = np.array(pe)

# 4. Dual-Buffer System
# agent_buffer: Stores real + imaginary transitions (Used to train Actor/Critic)
# rom_buffer: Stores ONLY real transitions (Used to train the Surrogate Model)
agent_buffer = DecentralizedReplayBuffer(max_size=200000, obs_dim=local_y_dim, action_dim=1, n_mu=z_dim)
rom_buffer = DecentralizedReplayBuffer(max_size=200000, obs_dim=local_y_dim, action_dim=1, n_mu=z_dim)

# --- JIT Training & Inference Functions ---

@jax.jit
def train_surrogate_step(surrogate_params, opt_surrogate, y, u, z, next_y):
    """Updates surrogate model to minimize prediction error."""
    mu = z[:, -n_mu:] 
    
    def loss_fn(p):
        pred_y = surrogate.apply(p, y, u, mu)
        return jnp.mean((pred_y - next_y)**2)
    
    loss, grads = jax.value_and_grad(loss_fn)(surrogate_params)
    updates, opt_surrogate = tx_surrogate.update(grads, opt_surrogate)
    return optax.apply_updates(surrogate_params, updates), opt_surrogate, loss

@jax.jit
def train_td3_step(actor_params, critic1_params, critic2_params, 
                   t_actor_params, t_critic1_params, t_critic2_params,
                   opt_actor, opt_c1, opt_c2, 
                   z, y, u, r, next_y, key):
    """Core TD3 update with Hypernetworks."""
    
    key, noise_key = jax.random.split(key)
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * 0.2, -0.5, 0.5)
    next_u = jnp.clip(actor.apply(t_actor_params, z, next_y) + noise, -5.0, 5.0)
    
    q1_target = critic.apply(t_critic1_params, z, next_y, next_u)
    q2_target = critic.apply(t_critic2_params, z, next_y, next_u)
    target_q = r + 0.99 * jnp.minimum(q1_target, q2_target)
    
    def q_loss_fn(p, q_net):
        q_pred = q_net.apply(p, z, y, u)
        return jnp.mean((q_pred - target_q)**2)
    
    loss_c1, grads_c1 = jax.value_and_grad(q_loss_fn)(critic1_params, critic)
    loss_c2, grads_c2 = jax.value_and_grad(q_loss_fn)(critic2_params, critic)
    
    up_c1, opt_c1 = tx_critic.update(grads_c1, opt_c1)
    up_c2, opt_c2 = tx_critic.update(grads_c2, opt_c2)
    critic1_params = optax.apply_updates(critic1_params, up_c1)
    critic2_params = optax.apply_updates(critic2_params, up_c2)
    
    def a_loss_fn(p):
        a_pred = actor.apply(p, z, y)
        q_pred = critic.apply(critic1_params, z, y, a_pred)
        return -jnp.mean(q_pred)
    
    loss_a, grads_a = jax.value_and_grad(a_loss_fn)(actor_params)
    up_a, opt_actor = tx_actor.update(grads_a, opt_actor)
    actor_params = optax.apply_updates(actor_params, up_a)
    
    tau = 0.005
    t_actor_params = jax.tree_util.tree_map(lambda x, y: tau*x + (1-tau)*y, actor_params, t_actor_params)
    t_critic1_params = jax.tree_util.tree_map(lambda x, y: tau*x + (1-tau)*y, critic1_params, t_critic1_params)
    t_critic2_params = jax.tree_util.tree_map(lambda x, y: tau*x + (1-tau)*y, critic2_params, t_critic2_params)
    
    return actor_params, critic1_params, critic2_params, t_actor_params, t_critic1_params, t_critic2_params, opt_actor, opt_c1, opt_c2, loss_a

@jax.jit
def predict_next_state(surrogate_params, y, u, z):
    mu = z[:, -n_mu:] 
    return surrogate.apply(surrogate_params, y, u, mu)

@jax.jit
def get_actor_actions(actor_params, z, y, noise_key=None):
    """Returns deterministic actions if noise_key is None, otherwise adds exploration noise."""
    action = actor.apply(actor_params, z, y)
    if noise_key is not None:
        action += jax.random.normal(noise_key, action.shape) * 0.1
    return jnp.clip(action, -5.0, 5.0)


# --- Main Loop ---
print(f"Starting Training: {'MB-HypeMARL' if USE_MB_HYPEMARL else 'HypeMARL'}")
start_time = time.time()

for episode in range(MAX_EPISODES):
    
    # ---------------------------------------------------------
    # 1. Noise-Free Evaluation Phase
    # ---------------------------------------------------------
    if episode % EVAL_INT == 0:
        eval_obs = env.reset(seed=EVAL_SEED + episode)
        eval_energy = 0.0
        
        for step_count in range(MAX_STEPS):
            eval_y = eval_obs[:, :local_y_dim]
            eval_mu = eval_obs[:, local_y_dim:]
            eval_z = np.concatenate([pe, eval_mu], axis=-1)
            
            # Deterministic action selection
            eval_action = np.array(get_actor_actions(actor_params, eval_z, eval_y))
            eval_obs, _, eval_done, eval_info = env.step(eval_action.flatten())
            
            # Extract tracking metric (-global_reward is the MSE Energy)
            eval_energy += -eval_info['global_reward']
            
            # Break if PDE blew up
            if eval_done:
                break
            
        # Average over actual steps taken
        mean_eval_energy = eval_energy / (step_count + 1)
        print(f"Eval Episode {episode:03d} | Mean System Energy: {mean_eval_energy:.6f}")

    # ---------------------------------------------------------
    # 2. Real Environment Interaction
    # ---------------------------------------------------------
    obs = env.reset(seed=int(time.time()) + episode)
    
    for t in range(MAX_STEPS):
        y = obs[:, :local_y_dim]
        mu = obs[:, local_y_dim:]
        z = np.concatenate([pe, mu], axis=-1)
        
        if episode < WARMUP_EPISODES:
            action = np.random.uniform(-5.0, 5.0, (N_AGENTS, 1))
        else:
            key, subkey = jax.random.split(key)
            action = np.array(get_actor_actions(actor_params, z, y, subkey))
            
        next_obs, rewards, done, info = env.step(action.flatten())
        
        # If the environment blew up, DO NOT store but abort the episode entirely to protect the buffer.
        if done and info.get('global_reward') == -100.0:
            break
        
        next_y = next_obs[:, :local_y_dim]
        
        # Add to BOTH buffers
        agent_buffer.add(y, action, rewards, next_y, z)
        rom_buffer.add(y, action, rewards, next_y, z)
        obs = next_obs
        
        # Train Surrogate on Real Data Only
        if USE_MB_HYPEMARL and rom_buffer.size > BATCH_SIZE:
            b_y, b_act, _, b_ny, b_z = rom_buffer.sample(BATCH_SIZE)
            surrogate_params, opt_surrogate, s_loss = train_surrogate_step(
                surrogate_params, opt_surrogate, b_y, b_act, b_z, b_ny
            )
            
        # Train TD3 on Agent Data
        if agent_buffer.size > BATCH_SIZE:
            b_y, b_act, b_rew, b_ny, b_z = agent_buffer.sample(BATCH_SIZE)
            key, subkey = jax.random.split(key)
            res = train_td3_step(actor_params, critic1_params, critic2_params, 
                                 target_actor_params, target_critic1_params, target_critic2_params,
                                 opt_actor, opt_critic1, opt_critic2, 
                                 b_z, b_y, b_act, b_rew, b_ny, subkey)
            
            actor_params, critic1_params, critic2_params = res[:3]
            target_actor_params, target_critic1_params, target_critic2_params = res[3:6]
            opt_actor, opt_critic1, opt_critic2 = res[6:9]

    # ---------------------------------------------------------
    # 3. MB-HypeMARL Imaginary Rollouts
    # ---------------------------------------------------------
    if USE_MB_HYPEMARL and episode >= WARMUP_EPISODES:
        for _ in range(IMAGINARY_RATIO):
            # Seed imagination using real states from the ROM buffer
            b_y, _, _, _, b_z = rom_buffer.sample(BATCH_SIZE)
            
            # Unroll for a limited horizon to prevent compounding prediction errors
            for _ in range(IMAGINATION_HORIZON):
                key, subkey = jax.random.split(key)
                b_act = np.array(get_actor_actions(actor_params, b_z, b_y, subkey))
                b_ny = np.array(predict_next_state(surrogate_params, b_y, b_act, b_z))
                
                if np.isnan(b_ny).any() or np.isinf(b_ny).any():
                    break # Abort imagination to protect the buffer
                
                # Extract the center of the error patch (index 10 out of 40)
                center_errors = b_ny[:, 10]
                b_rew = -np.square(center_errors).reshape(-1, 1)
                # ---------------
                
                # Add synthetic transitions ONLY to the agent buffer
                agent_buffer.add(b_y, b_act, b_rew, b_ny, b_z)
                b_y = b_ny
                
                # Train TD3 on mixed buffer (Real + Imaginary)
                if agent_buffer.size > BATCH_SIZE:
                    s_y, s_act, s_rew, s_ny, s_z = agent_buffer.sample(BATCH_SIZE)
                    key, subkey = jax.random.split(key)
                    res = train_td3_step(actor_params, critic1_params, critic2_params, 
                                         target_actor_params, target_critic1_params, target_critic2_params,
                                         opt_actor, opt_critic1, opt_critic2, 
                                         s_z, s_y, s_act, s_rew, s_ny, subkey)
                    actor_params, critic1_params, critic2_params = res[:3]
                    target_actor_params, target_critic1_params, target_critic2_params = res[3:6]
                    opt_actor, opt_critic1, opt_critic2 = res[6:9]

# --- Save Weights ---
print(f"Training finished in {time.time() - start_time:.2f}s. Saving parameters...")
save_dict = {
    'actor': actor_params,
    'critic1': critic1_params,
    'critic2': critic2_params,
    'surrogate': surrogate_params
}

with open('models/hypemarl_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes(save_dict))
print("Weights saved successfully to hypemarl_params.msgpack.")