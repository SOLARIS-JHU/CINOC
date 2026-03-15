import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.serialization
import numpy as np
import time
import pickle
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from examples.ks2d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks2d.decentralized.dynamics_dual import PDEDynamics2D 
from models_ppo import PPOActor2DKS, PPOCritic2DKS, U_MAX

# --- Configurations ---
N_AGENTS = 100
L_DOMAIN = 32.0
N_GRID = 64

# KS2D Specific Control Timing
MAX_ENV_STEPS = 50     # Control steps (T_steps)
SUBSTEPS = 10          # Physics steps per control step
DT = 0.005             # Physics dt
NUM_PARALLEL_ENVS = 64

# PPO Specific Configs
ROLLOUT_STEPS = 50     # Usually aligned with max env steps for on-policy sync
PPO_EPOCHS = 4
MINIBATCH_SIZE = 128
TOTAL_UPDATES = 50#10000
EVAL_INT = 10 

key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics2D(policy_apply_fn=direct_control_policy)

# 2D Actuator Grid Setup (Fixed Positions)
grid_dim = int(np.sqrt(N_AGENTS))
x_lin = np.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + (L_DOMAIN/grid_dim)/2
xv, yv = jnp.meshgrid(x_lin, x_lin)
agent_positions = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
xi_fixed = jnp.array(agent_positions)
target_state = jnp.zeros((N_GRID, N_GRID))

# --- Gaussian Action Utils ---
def get_logprob_and_action(mean, log_std, key=None, action=None):
    std = jnp.exp(log_std)
    if action is None:
        noise = jax.random.normal(key, mean.shape)
        action = mean + noise * std
        action = jnp.clip(action, -U_MAX, U_MAX)
    
    var = std ** 2
    log_prob = -0.5 * ((action - mean) ** 2) / var - log_std - 0.5 * jnp.log(2 * jnp.pi)
    return action, jnp.sum(log_prob, axis=-1)

# --- Initialization ---
actor = PPOActor2DKS(n_agents=N_AGENTS)
critic = PPOCritic2DKS()

key, *subkeys = jax.random.split(key, 4)
dummy_u = jnp.zeros((1, N_GRID, N_GRID))

actor_params = actor.init(subkeys[0], dummy_u)
critic_params = critic.init(subkeys[1], dummy_u)

tx_actor = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
tx_critic = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-3))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- JAX-Native GAE ---
@jax.jit
def compute_gae_jax(rewards, values, dones, true_next_values, last_val, gamma=0.99, lam=0.95):
    def scan_fn(carry, transition):
        r, v, d, true_next_v = transition
        gae, _ = carry
        delta = r + gamma * true_next_v * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        return (gae, v), gae
    
    _, advantages = jax.lax.scan(
        scan_fn, 
        (jnp.zeros_like(last_val), last_val), 
        (rewards, values, dones, true_next_values), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

# --- Parallel Physics Step (2D) ---
def parallel_physics_step(u_batch, actions, xi_fixed):
    def single_physics_step(u_s, act_s):
        traj = dynamics.unroll_controlled(
            u_init=u_s, xi_fixed=xi_fixed, u_target=target_state, params=act_s, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        return traj[0][-1]
    
    next_u_batch = jax.vmap(single_physics_step)(u_batch, actions)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_u_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_u_batch), axis=(1, 2)) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_u = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_u_batch), next_u_batch)
    
    # Centralized Global Energy Reward
    global_energy = jnp.mean(jnp.square(safe_u), axis=(1, 2))
    rewards_batch = -global_energy
    
    return safe_u, rewards_batch, dones_batch

# --- Minibatch Update Logic ---
def update_ppo_minibatch(a_params, c_params, opt_a, opt_c, b_u, b_a, b_logp, b_ret, b_adv):
    def loss_fn(ap, cp):
        mean, log_std = actor.apply(ap, b_u)
        _, new_logp = get_logprob_and_action(mean, log_std, action=b_a)
        
        entropy = jnp.sum(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=-1).mean()
        
        ratio = jnp.exp(new_logp - b_logp)
        pg_loss1 = -b_adv * ratio
        pg_loss2 = -b_adv * jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean() - 0.01 * entropy
        
        values = critic.apply(cp, b_u).squeeze(-1)
        critic_loss = 0.5 * jnp.mean((b_ret - values) ** 2)
        
        return actor_loss + 0.5 * critic_loss, (actor_loss, critic_loss, entropy)

    (total_loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(a_params, c_params)
    
    up_a, opt_a = tx_actor.update(grads[0], opt_a)
    up_c, opt_c = tx_critic.update(grads[1], opt_c)
    
    return optax.apply_updates(a_params, up_a), optax.apply_updates(c_params, up_c), opt_a, opt_c, metrics

# --- THE PURE JAX PPO TRAIN STEP ---
def train_step(runner_state, state_bank):
    a_params, c_params, opt_a, opt_c, u_batch, env_counts, rng = runner_state
    
    # 1. Rollout Phase
    def _env_step(carry, _):
        u, counts, k = carry
        k, act_k, reset_k = jax.random.split(k, 3)
        
        mean, log_std = actor.apply(a_params, u)
        actions, log_probs = get_logprob_and_action(mean, log_std, key=act_k)
        values = critic.apply(c_params, u).squeeze(-1)
        
        next_u, rewards, dones = parallel_physics_step(u, actions, xi_fixed)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones, truncs)
        
        fresh_states = jax.random.choice(reset_k, state_bank, shape=(NUM_PARALLEL_ENVS,))
        
        # Get the value of the true next state BEFORE overwriting it with resets
        true_next_v = critic.apply(c_params, next_u).squeeze(-1)

        next_u_final = jnp.where(needs_reset[:, None, None], fresh_states, next_u)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        transition = (u, actions, rewards, values, log_probs, dones, true_next_v)
        return (next_u_final, next_counts, k), transition

    carry = (u_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_u_batch, next_env_counts, rng) = carry
    t_u, t_a, t_r, t_v, t_logp, t_d, t_true_next_v = transitions
    
    # 2. GAE Phase
    last_val = critic.apply(c_params, next_u_batch).squeeze(-1)
    adv, ret = compute_gae_jax(t_r, t_v, t_d, t_true_next_v, last_val)
    
    # Flatten across time and envs (preserving grid dims for state)
    f_u = t_u.reshape(-1, N_GRID, N_GRID)
    f_a = t_a.reshape(-1, N_AGENTS)
    f_logp = t_logp.reshape(-1) 
    f_ret = ret.reshape(-1)
    f_adv = adv.reshape(-1)
    
    # Normalize advantages across the ENTIRE batch here
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
    dataset_size = f_u.shape[0]

    # 3. Optimization Phase
    def _update_epoch(epoch_carry, _):
        ap, cp, oa, oc, k = epoch_carry
        k, subk = jax.random.split(k)
        indices = jax.random.permutation(subk, dataset_size)
        
        def _update_minibatch(mb_carry, start_idx):
            ap_, cp_, oa_, oc_ = mb_carry
            batch_idx = jax.lax.dynamic_slice(indices, (start_idx,), (MINIBATCH_SIZE,))
            
            ap_n, cp_n, oa_n, oc_n, metrics = update_ppo_minibatch(
                ap_, cp_, oa_, oc_, 
                f_u[batch_idx], f_a[batch_idx], f_logp[batch_idx], 
                f_ret[batch_idx], f_adv[batch_idx]
            )
            return (ap_n, cp_n, oa_n, oc_n), metrics
            
        mb_starts = jnp.arange(0, dataset_size, MINIBATCH_SIZE)
        (ap, cp, oa, oc), epoch_metrics = jax.lax.scan(_update_minibatch, (ap, cp, oa, oc), mb_starts)
        return (ap, cp, oa, oc, k), epoch_metrics

    epoch_carry = (a_params, c_params, opt_a, opt_c, rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (a_params, c_params, opt_a, opt_c, rng) = epoch_carry

    new_runner_state = (a_params, c_params, opt_a, opt_c, next_u_batch, next_env_counts, rng)
    
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[0].mean(),
        "critic_loss": ppo_metrics[1].mean()
    }
    
    return new_runner_state, metrics

# --- SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(runner_state, state_bank):
    def scan_step(carry, _):
        new_state, metrics = train_step(carry, state_bank)
        return new_state, metrics
    
    return jax.lax.scan(scan_step, runner_state, None, length=EVAL_INT)

# --- Fast Evaluation ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(a_params, init_u, max_steps):
    def step_fn(state, _):
        mean, _ = actor.apply(a_params, state[None, ...])
        act_flat = mean.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            u_init=state, xi_fixed=xi_fixed, u_target=target_state, params=act_flat, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        next_state = traj[0][-1]
        
        energy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 100.0)
        
        return next_state, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_u, None, length=max_steps)
    return jnp.mean(energies), jnp.any(crashes)

# --- Python Execution Loop ---
print("Loading 2D KS Initial Conditions...")
data_dir = Path('../../data')
data_dir.mkdir(parents=True, exist_ok=True)
file_path = data_dir / 'ks2d_chaotic_ics_64.pkl'

if file_path.exists():
    with open(file_path, 'rb') as f:
        state_bank = jnp.array(pickle.load(f))
    print(f"Loaded {len(state_bank)} ICs from {file_path}")
else:
    print("Generating ICs (this may take a few minutes for KS2D)...")
    state_bank = get_batch_initial_conditions(key, 500, N_GRID, L_DOMAIN)
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(state_bank), f)

key, subkey = jax.random.split(key)
u_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))

initial_runner_state = (
    actor_params, critic_params, opt_actor, opt_critic, 
    u_batch, jnp.zeros(NUM_PARALLEL_ENVS), key
)

print("Starting Massively Parallel Pure JAX PPO Training (Chunked 2D)...")
start_time = time.time()

runner_state = initial_runner_state
num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk in trange(num_chunks):
    current_update = chunk * EVAL_INT
    
    # Evaluate at the start of the chunk
    eval_u = state_bank[0]
    eval_energy, crashed = fast_eval_episode(runner_state[0], eval_u, MAX_ENV_STEPS)
    
    status = "[CRASHED]" if crashed else f"{eval_energy:.6f}"
    print(f"Update {current_update:04d} | Eval Energy: {status} | Time: {time.time()-start_time:.1f}s")

    # Run the compiled chunk
    runner_state, batch_metrics = train_chunk(runner_state, state_bank)

# Save output
actor_params_final = runner_state[0]
with open('models/ppo_ks2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params_final}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")