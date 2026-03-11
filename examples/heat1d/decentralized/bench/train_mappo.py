import jax
import jax.numpy as jnp
import optax
import flax.serialization
from flax.training.train_state import TrainState
import numpy as np
import time
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from env_he import HeatHypeMARLEnv 
from utils_hypemarl import get_sinusoidal_encoding
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics 
from models_mappo import MAPPOActor, MAPPOCritic 

# --- PPO Configurations ---
N_AGENTS = 8 
L_DOMAIN = 1.0
N_GRID = 100

NUM_PARALLEL_ENVS = 256
ROLLOUT_STEPS = 300      # Match episode length
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1024
TOTAL_TIMESTEPS = 50_000_000

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    u = action_params[:, 0]
    v = action_params[:, 1]
    return u, v

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = HeatHypeMARLEnv(dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=ROLLOUT_STEPS)

pe_dim = 2048
stored_obs_dim = env.local_y_dim + env.n_mu 
total_input_dim = stored_obs_dim + pe_dim
mu_jax = jnp.array(env.mu)
window_size = env.window_size

# --- Helper: On-The-Fly Positional Encoding ---
@jax.jit
def attach_pe_batch(obs_no_pe, xi_batch):
    """Generates the 2048-dim PE on the fly to save VRAM."""
    def single_env_pe(obs_env, xi_env):
        pe = get_sinusoidal_encoding(xi_env, d=pe_dim)
        return jnp.concatenate([obs_env, pe], axis=-1)
    return jax.vmap(single_env_pe)(obs_no_pe, xi_batch)

# --- PPO Core Functions ---
@jax.jit
def get_action_and_value(actor_params, critic_params, obs_no_pe, z, target, xi, key):
    full_obs = attach_pe_batch(obs_no_pe, xi)
    
    mean, log_std = actor.apply(actor_params, full_obs)
    std = jnp.exp(log_std)
    
    action = mean + std * jax.random.normal(key, mean.shape)
    log_prob = -0.5 * jnp.sum(jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    
    val = critic.apply(critic_params, z, target, xi)
    
    return action, log_prob, val

@jax.jit
def get_value(critic_params, z, target, xi):
    return critic.apply(critic_params, z, target, xi)

@jax.jit
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    advantages = jnp.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(ROLLOUT_STEPS)):
        if t == ROLLOUT_STEPS - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t + 1]
            
        delta = rewards[t] + gamma * nextvalues * nextnonterminal[:, None] - values[t]
        advantages = advantages.at[t].set(delta + gamma * lam * nextnonterminal[:, None] * lastgaelam)
        lastgaelam = advantages[t]
        
    returns = advantages + values
    return advantages, returns

# --- PPO Update Step ---
@jax.jit
def ppo_update_epoch(actor_state, critic_state, batch):
    obs_no_pe, z, target, xi, actions, old_log_probs, advantages, returns = batch
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def actor_loss_fn(params):
        full_obs = attach_pe_batch(obs_no_pe, xi)
        mean, log_std = actor.apply(params, full_obs)
        std = jnp.exp(log_std)
        
        log_probs = -0.5 * jnp.sum(jnp.square((actions - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)
        ratio = jnp.exp(log_probs - old_log_probs)
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        
        entropy_loss = entropy.mean()
        return pg_loss - ENTROPY_COEF * entropy_loss, (pg_loss, entropy_loss)
        
    def value_loss_fn(params):
        v = critic.apply(params, z, target, xi)
        return jnp.mean(jnp.square(v - returns))
        
    (a_loss, aux), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
    v_loss, v_grads = jax.value_and_grad(value_loss_fn)(critic_state.params)
    
    actor_state = actor_state.apply_gradients(grads=a_grads)
    critic_state = critic_state.apply_gradients(grads=v_grads)
    
    return actor_state, critic_state, a_loss, v_loss, aux[1]

# --- Environment Handlers ---
@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_jax(full_state, target_st, xi_n, window_size):
    error = full_state - target_st
    error_grad = jnp.gradient(error)
    n_pde = full_state.shape[0]
    half_window = window_size // 2

    padded_error = jnp.pad(error, (half_window, half_window), mode='constant', constant_values=0.0)
    padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='constant', constant_values=0.0)

    def get_local_obs(xi):
        center_idx = jax.lax.stop_gradient((xi * (n_pde - 1)).astype(int)) + half_window
        start = center_idx - half_window
        p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
        p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
        p_err = jax.image.resize(p_err, (20,), method='bilinear')
        p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
        return jnp.concatenate([p_err, p_grad])

    return jax.vmap(get_local_obs)(xi_n)

@jax.jit
def build_marl_obs_batch(z_batch, target_batch, xi_batch):
    def single_env_obs(state, target, xi):
        y_local = extract_patches_jax(state, target, xi, window_size)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return jnp.concatenate([y_local, mu_broadcast], axis=-1)
    return jax.vmap(single_env_obs)(z_batch, target_batch, xi_batch)

@jax.jit
def parallel_marl_physics_step(z_batch, xi_batch, target_batch, actions, key):
    keys = jax.random.split(key, z_batch.shape[0])
    
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, 
            t_steps=1,
        )
        return traj[0][-1], traj[1][-1]
    
    next_z_batch, next_xi_batch = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_z_batch).all(axis=-1, keepdims=True))
    dones_batch = is_invalid
    
    safe_z = jnp.where(dones_batch, jnp.zeros_like(next_z_batch), next_z_batch)
    safe_xi = jnp.where(dones_batch, xi_batch, next_xi_batch)
    
    next_obs_batch_no_pe = build_marl_obs_batch(safe_z, target_batch, safe_xi)
    
    u_batch = actions[..., 0]
    v_batch = actions[..., 1]
    
    center_errors = next_obs_batch_no_pe[:, :, 10]
    r_track = -5.0 * jnp.square(center_errors)
    r_effort = -0.001 * (jnp.square(u_batch) + 0.1 * jnp.square(v_batch))
    margin = 0.02
    r_bound = -100.0 * (jnp.maximum(0.0, margin - safe_xi)**2 + jnp.maximum(0.0, safe_xi - (1.0 - margin))**2)
    R_safe = 0.05
    dists = jnp.abs(safe_xi[:, :, None] - safe_xi[:, None, :])
    mask = jnp.eye(N_AGENTS)[None, :, :]
    r_coll = -1.0 * jnp.sum(jnp.maximum(0.0, R_safe - (dists + mask * 1.0)) ** 2, axis=2)
    
    rewards_batch = r_track + r_effort + r_bound + r_coll
    rewards_batch = jnp.where(dones_batch, -100.0, rewards_batch)
    
    return safe_z, safe_xi, next_obs_batch_no_pe, rewards_batch, dones_batch.squeeze(-1)

# --- Fast JIT-Compiled Rollout Loop ---
@partial(jax.jit, static_argnames=['rollout_steps'])
def collect_rollout(actor_params, critic_params, init_z, init_xi, init_target, init_obs, key, rollout_steps, z_bank, target_bank, xi_single):
    def step_fn(state, _):
        z, xi, target, obs_no_pe, rng = state
        rng, act_rng, phys_rng, reset_rng = jax.random.split(rng, 4)

        action, log_prob, val = get_action_and_value(actor_params, critic_params, obs_no_pe, z, target, xi, act_rng)
        next_z, next_xi, next_obs_no_pe, reward, done = parallel_marl_physics_step(z, xi, target, action, phys_rng)

        idx_reset = jax.random.randint(reset_rng, (NUM_PARALLEL_ENVS,), 0, 1000)
        fresh_z = z_bank[idx_reset]
        fresh_target = target_bank[idx_reset]
        fresh_xi = jnp.tile(xi_single, (NUM_PARALLEL_ENVS, 1))

        z_new = jnp.where(done[:, None], fresh_z, next_z)
        target_new = jnp.where(done[:, None], fresh_target, target)
        xi_new = jnp.where(done[:, None], fresh_xi, next_xi)

        obs_new = jnp.where(
            done[:, None, None], 
            build_marl_obs_batch(z_new, target_new, xi_new), 
            next_obs_no_pe
        )

        transition = (obs_no_pe, z, target, xi, action, log_prob, reward, val, done)
        return (z_new, xi_new, target_new, obs_new, rng), transition

    final_state, transitions = jax.lax.scan(
        step_fn, 
        (init_z, init_xi, init_target, init_obs, key), 
        None, 
        length=rollout_steps
    )
    return final_state, transitions

# --- FAST EVALUATION LOGIC ---
@jax.jit
def get_eval_action(actor_params, obs_no_pe, xi_batch):
    """Deterministic action for evaluation (uses mean, no sampling)."""
    full_obs = attach_pe_batch(obs_no_pe, xi_batch)
    mean, _ = actor.apply(actor_params, full_obs)
    return mean

@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_z, init_xi, target_z, max_steps):
    """Evaluates the actor deterministically for a full episode."""
    def step_fn(state_tuple, _):
        z_curr, xi_curr = state_tuple
        
        # Add batch dimension for the observation builder
        obs_no_pe = build_marl_obs_batch(z_curr[None, ...], target_z[None, ...], xi_curr[None, ...]) 
        
        # Get deterministic action
        act = get_eval_action(actor_params, obs_no_pe, xi_curr[None, ...])
        act_flat = act.squeeze(0)
        
        # Step physics
        traj = dynamics.unroll_controlled(
            z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=act_flat, 
            t_steps=1
        )
        next_z, next_xi = traj[0][-1], traj[1][-1]
        
        # Compute tracking MSE
        mse = jnp.mean((next_z - target_z)**2)
        crashed = jnp.isnan(next_z).any() | jnp.isinf(next_z).any()
        
        return (next_z, next_xi), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# --- Initialize States & Networks ---
actor = MAPPOActor(n_agents=N_AGENTS)
critic = MAPPOCritic(n_agents=N_AGENTS)

key, act_k, val_k = jax.random.split(key, 3)

dummy_obs_full = jnp.zeros((1, N_AGENTS, total_input_dim))
dummy_z = jnp.zeros((1, N_GRID))
dummy_target = jnp.zeros((1, N_GRID))
dummy_xi = jnp.zeros((1, N_AGENTS))

actor_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor.init(act_k, dummy_obs_full),
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LR, eps=1e-5))
)

critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic.init(val_k, dummy_z, dummy_target, dummy_xi),
    tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LR, eps=1e-5))
)

# --- Data Generation ---
print("Pre-generating starting state & target banks...")
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.4))(bank_keys)
xi_init_single = jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32)

num_updates = TOTAL_TIMESTEPS // (NUM_PARALLEL_ENVS * ROLLOUT_STEPS)

# Setup initial rollout state
key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, 1000)
z_batch = z_init_bank[idx]
target_batch = z_target_bank[idx]
xi_batch = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))
obs_batch = build_marl_obs_batch(z_batch, target_batch, xi_batch)

print(f"Starting MAPPO Training for {num_updates} Rollout Generations...")
start_time = time.time()

for update in trange(num_updates):
    # --- PHASE 1: JIT-Compiled Trajectory Collection ---
    final_state, transitions = collect_rollout(
        actor_state.params, critic_state.params, 
        z_batch, xi_batch, target_batch, obs_batch, key, 
        ROLLOUT_STEPS, z_init_bank, z_target_bank, xi_init_single
    )
    
    b_obs, b_z, b_target, b_xi, b_acts, b_logprobs, b_rewards, b_values, b_dones = transitions
    z_batch, xi_batch, target_batch, obs_batch, key = final_state

    # --- PHASE 2: Advantage Calculation ---
    next_val = get_value(critic_state.params, z_batch, target_batch, xi_batch)
    advantages, returns = compute_gae(b_rewards, b_values, b_dones, next_val, GAMMA, GAE_LAMBDA)
    
    flat_obs = b_obs.reshape(-1, N_AGENTS, stored_obs_dim)  
    flat_z = b_z.reshape(-1, N_GRID)
    flat_target = b_target.reshape(-1, N_GRID)
    flat_xi = b_xi.reshape(-1, N_AGENTS)
    
    flat_acts = b_acts.reshape(-1, N_AGENTS, 2)
    flat_logprobs = b_logprobs.reshape(-1, N_AGENTS)
    flat_advs = advantages.reshape(-1, N_AGENTS)
    flat_rets = returns.reshape(-1, N_AGENTS)
    
    # --- PHASE 3: PPO Epoch Optimization ---
    dataset_size = flat_obs.shape[0]  
    indices = np.arange(dataset_size)
    
    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        
        for start in range(0, dataset_size, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_idx = indices[start:end]
            
            mb_batch = (
                flat_obs[mb_idx], flat_z[mb_idx], flat_target[mb_idx], flat_xi[mb_idx],
                flat_acts[mb_idx], flat_logprobs[mb_idx], flat_advs[mb_idx], flat_rets[mb_idx]
            )
            
            actor_state, critic_state, al, vl, ent = ppo_update_epoch(actor_state, critic_state, mb_batch)

    # --- EVALUATION AND LOGGING ---
    if update % 10 == 0:
        eval_z = z_init_bank[0] 
        eval_target = z_target_bank[0]
        eval_xi = xi_init_single
        
        eval_e, crashed = fast_eval_episode(actor_state.params, eval_z, eval_xi, eval_target, ROLLOUT_STEPS)
        
        avg_reward = b_rewards.sum(axis=0).mean()
        status = "[CRASHED]" if crashed else f"{eval_e:.6f}"
        
        print(f"Update {update:04d} | Ret: {avg_reward:7.2f} | Eval MSE: {status} | Val L: {vl:.4f} | Act L: {al:.4f} | Ent: {ent:.4f}")

# Save output
with open('models/mappo_heat_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_state.params, 'critic': critic_state.params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")