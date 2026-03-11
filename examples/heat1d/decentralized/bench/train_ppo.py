import jax
import jax.numpy as jnp
import optax
import flax.serialization
import time
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from models_ppo import PPOActor, PPOCritic, U_MAX, V_MAX
from ppo_utils import get_logprob_and_action
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 8
N_GRID = 100
MAX_ENV_STEPS = 300
NUM_PARALLEL_ENVS = 256

# PPO Specific Configs
ROLLOUT_STEPS = 128
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1024
TOTAL_UPDATES = 25000
EVAL_INT = 10 

key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params[:, 0], action_params[:, 1]

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)

# --- Initialization ---
actor = PPOActor(n_agents=N_AGENTS)
critic = PPOCritic()

key, *subkeys = jax.random.split(key, 4)
dummy_z = jnp.zeros((1, N_GRID))
dummy_target = jnp.zeros((1, N_GRID))
dummy_xi = jnp.zeros((1, N_AGENTS))

actor_params = actor.init(subkeys[0], dummy_z, dummy_target, dummy_xi)
critic_params = critic.init(subkeys[1], dummy_z, dummy_target, dummy_xi)

tx_actor = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
tx_critic = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-3))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- Pre-generate Starting State Banks ---
# We do this globally so the JIT compiled function can access them for resets
print("Pre-generating starting state & target banks...")
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.4))(bank_keys)
xi_init_single = jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32)

# --- JAX-Native GAE ---
# Using lax.scan instead of reversed Python loops prevents massive trace times
@jax.jit
def compute_gae_jax(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    def scan_fn(carry, transition):
        r, v, d = transition
        gae, next_v = carry
        delta = r + gamma * next_v * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        return (gae, v), gae
    
    _, advantages = jax.lax.scan(
        scan_fn, 
        (jnp.zeros_like(next_value), next_value), 
        (rewards, values, dones), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

# --- Parallel Physics Step ---
def parallel_physics_step(z_batch, xi_batch, target_batch, actions, phys_key):
    keys = jax.random.split(phys_key, z_batch.shape[0])
    
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, t_steps=1
        )
        return traj[0][-1], traj[1][-1]
    
    next_z, next_xi = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    dones = jnp.logical_not(jnp.isfinite(next_z).all(axis=-1, keepdims=True))
    
    safe_z = jnp.where(dones, jnp.zeros_like(next_z), next_z)
    safe_xi = jnp.where(dones, xi_batch, next_xi)
    
    mse = jnp.mean((safe_z - target_batch)**2, axis=-1, keepdims=True)
    effort = jnp.mean(0.001 * (jnp.square(actions[..., 0]) + 0.1 * jnp.square(actions[..., 1])), axis=-1, keepdims=True)
    
    margin = 0.02
    oob_penalty = 100.0 * (jnp.maximum(0.0, margin - safe_xi)**2 + jnp.maximum(0.0, safe_xi - (1.0 - margin))**2)
    mean_oob = jnp.mean(oob_penalty, axis=-1, keepdims=True)
    
    R_safe = 0.05
    dists = jnp.abs(safe_xi[:, :, None] - safe_xi[:, None, :])
    mask = jnp.eye(N_AGENTS)[None, :, :]
    coll_penalty = 1.0 * jnp.sum(jnp.maximum(0.0, R_safe - (dists + mask * 1.0)) ** 2, axis=2)
    mean_coll = jnp.mean(coll_penalty, axis=-1, keepdims=True)
    
    rewards = -mse - effort - mean_oob - mean_coll
    return safe_z, safe_xi, rewards, dones

# --- Minibatch Update Logic ---
def update_ppo_minibatch(a_params, c_params, opt_a, opt_c, b_z, b_zt, b_xi, b_a, b_logp, b_ret, b_adv):
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    def loss_fn(ap, cp):
        mean, log_std = actor.apply(ap, b_z, b_zt, b_xi)
        _, new_logp = get_logprob_and_action(mean, log_std, action=b_a)
        
        entropy = jnp.sum(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=(-1, -2)).mean()
        
        ratio = jnp.exp(new_logp - b_logp)
        pg_loss1 = -b_adv * ratio
        pg_loss2 = -b_adv * jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean() - 0.01 * entropy
        
        values = critic.apply(cp, b_z, b_zt, b_xi).squeeze(-1)
        critic_loss = 0.5 * jnp.mean((b_ret - values) ** 2)
        
        return actor_loss + 0.5 * critic_loss, (actor_loss, critic_loss, entropy)

    (total_loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(a_params, c_params)
    
    up_a, opt_a = tx_actor.update(grads[0], opt_a)
    up_c, opt_c = tx_critic.update(grads[1], opt_c)
    
    return optax.apply_updates(a_params, up_a), optax.apply_updates(c_params, up_c), opt_a, opt_c, metrics

# --- THE PURE JAX PPO TRAIN STEP ---
@jax.jit
def train_step(runner_state):
    """Executes a full rollout AND all optimization epochs entirely on the GPU."""
    a_params, c_params, opt_a, opt_c, z_batch, target_batch, xi_batch, env_counts, rng = runner_state
    
    # 1. Rollout Phase (Compiled loop)
    def _env_step(carry, _):
        z, zt, xi, counts, k = carry
        k, act_k, phys_k, reset_k = jax.random.split(k, 4)
        
        mean, log_std = actor.apply(a_params, z, zt, xi)
        actions, log_probs = get_logprob_and_action(mean, log_std, key=act_k)
        values = critic.apply(c_params, z, zt, xi).squeeze(-1)
        
        next_z, next_xi, rewards, dones = parallel_physics_step(z, xi, zt, actions, phys_k)
        
        counts += 1
        truncs = counts >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        idx_reset = jax.random.randint(reset_k, (NUM_PARALLEL_ENVS,), 0, 1000)
        fresh_z = z_init_bank[idx_reset]
        fresh_target = z_target_bank[idx_reset]
        fresh_xi = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))
        
        next_z = jnp.where(needs_reset[:, None], fresh_z, next_z)
        next_zt = jnp.where(needs_reset[:, None], fresh_target, zt)
        next_xi = jnp.where(needs_reset[:, None], fresh_xi, next_xi)
        next_counts = jnp.where(needs_reset, 0, counts)
        
        transition = (z, zt, xi, actions, rewards.squeeze(-1), values, log_probs, dones.squeeze(-1))
        return (next_z, next_zt, next_xi, next_counts, k), transition

    carry = (z_batch, target_batch, xi_batch, env_counts, rng)
    carry, transitions = jax.lax.scan(_env_step, carry, None, length=ROLLOUT_STEPS)
    (next_z_batch, next_target_batch, next_xi_batch, next_env_counts, rng) = carry
    t_z, t_zt, t_xi, t_a, t_r, t_v, t_logp, t_d = transitions
    
    # 2. GAE Phase
    next_value = critic.apply(c_params, next_z_batch, next_target_batch, next_xi_batch).squeeze(-1)
    adv, ret = compute_gae_jax(t_r, t_v, t_d, next_value)
    
    # Flatten across time and envs
    f_z = t_z.reshape(-1, N_GRID)
    f_zt = t_zt.reshape(-1, N_GRID)
    f_xi = t_xi.reshape(-1, N_AGENTS)
    f_a = t_a.reshape(-1, N_AGENTS, 2)
    f_logp = t_logp.reshape(-1) 
    f_ret = ret.reshape(-1)
    f_adv = adv.reshape(-1)
    
    dataset_size = f_z.shape[0]

    # 3. Optimization Phase (Compiled nested loops)
    def _update_epoch(epoch_carry, _):
        ap, cp, oa, oc, k = epoch_carry
        k, subk = jax.random.split(k)
        indices = jax.random.permutation(subk, dataset_size)
        
        def _update_minibatch(mb_carry, start_idx):
            ap_, cp_, oa_, oc_ = mb_carry
            batch_idx = jax.lax.dynamic_slice(indices, (start_idx,), (MINIBATCH_SIZE,))
            
            ap_n, cp_n, oa_n, oc_n, metrics = update_ppo_minibatch(
                ap_, cp_, oa_, oc_, 
                f_z[batch_idx], f_zt[batch_idx], f_xi[batch_idx], 
                f_a[batch_idx], f_logp[batch_idx], f_ret[batch_idx], f_adv[batch_idx]
            )
            return (ap_n, cp_n, oa_n, oc_n), metrics
            
        mb_starts = jnp.arange(0, dataset_size, MINIBATCH_SIZE)
        (ap, cp, oa, oc), epoch_metrics = jax.lax.scan(_update_minibatch, (ap, cp, oa, oc), mb_starts)
        return (ap, cp, oa, oc, k), epoch_metrics

    epoch_carry = (a_params, c_params, opt_a, opt_c, rng)
    epoch_carry, ppo_metrics = jax.lax.scan(_update_epoch, epoch_carry, None, length=PPO_EPOCHS)
    (a_params, c_params, opt_a, opt_c, rng) = epoch_carry

    new_runner_state = (a_params, c_params, opt_a, opt_c, next_z_batch, next_target_batch, next_xi_batch, next_env_counts, rng)
    
    # Return metrics (taking the mean over minibatches and epochs)
    metrics = {
        "mean_return": t_r.sum(axis=0).mean(),
        "actor_loss": ppo_metrics[0].mean(),
        "critic_loss": ppo_metrics[1].mean()
    }
    
    return new_runner_state, metrics

# --- Fast Evaluation ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(a_params, init_z, init_xi, target_z, max_steps):
    def step_fn(state, _):
        z_curr, xi_curr = state
        # In eval, we use deterministic mean
        mean, _ = actor.apply(a_params, z_curr[None, ...], target_z[None, ...], xi_curr[None, ...])
        act_flat = mean.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=act_flat, t_steps=1
        )
        next_z, next_xi = traj[0][-1], traj[1][-1]
        
        mse = jnp.mean((next_z - target_z)**2)
        crashed = jnp.isnan(next_z).any() | jnp.isinf(next_z).any()
        return (next_z, next_xi), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# --- Python Execution Loop ---
key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, 1000)

initial_runner_state = (
    actor_params, critic_params, opt_actor, opt_critic, 
    z_init_bank[idx], z_target_bank[idx], jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1)), 
    jnp.zeros(NUM_PARALLEL_ENVS), key
)

print("Starting Massively Parallel Pure JAX PPO Training...")
start_time = time.time()

runner_state = initial_runner_state

# This python loop now purely exists to print telemetry; the heavy lifting is completely internal
for update in trange(TOTAL_UPDATES):
    
    if update % EVAL_INT == 0:
        eval_z, eval_target, eval_xi = z_init_bank[0], z_target_bank[0], xi_init_single
        eval_mse, crashed = fast_eval_episode(runner_state[0], eval_z, eval_xi, eval_target, MAX_ENV_STEPS)
        
        status = "[CRASHED]" if crashed else f"{eval_mse:.6f}"
        print(f"Update {update:04d} | Eval Tracking MSE: {status} | Time: {time.time()-start_time:.1f}s")

    # One line executes the entire rollout + GAE + 4 epochs + 32 minibatches!
    runner_state, metrics = train_step(runner_state)

# Save output
actor_params_final = runner_state[0]
with open('models/ppo_heat_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params_final}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")