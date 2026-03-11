import jax
import jax.numpy as jnp
import optax
import flax.serialization
from flax import struct
import numpy as np
import time
from pathlib import Path
import sys
from functools import partial

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from env_he import HeatHypeMARLEnv 
from utils_hypemarl import get_sinusoidal_encoding
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics 
from models_hypemarl import HyperActor, HyperCritic, SurrogateModel, U_MAX, V_MAX

# --- Configurations ---
USE_MB_HYPEMARL = False
N_AGENTS = 8
L_DOMAIN = 1.0
N_GRID = 100
ENV_BATCH_SIZE = 256 
EVAL_INT = 500
POLICY_DELAY = 2 
MAX_ENV_STEPS = 300
NUM_PARALLEL_ENVS = 256
TOTAL_UPDATES = 10000 
WARMUP_UPDATES = 500

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    u = action_params[:, 0]
    v = action_params[:, 1]
    return u, v

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
env = HeatHypeMARLEnv(dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=MAX_ENV_STEPS)

local_y_dim = env.local_y_dim
pe_dim = 2048
z_dim = pe_dim 
stored_obs_dim = local_y_dim 
window_size = env.window_size

# Models
actor = HyperActor()
critic = HyperCritic()
surrogate = SurrogateModel()

key, *subkeys = jax.random.split(key, 6)
dummy_z = jnp.zeros((z_dim,))
dummy_y = jnp.zeros((local_y_dim,))
dummy_act = jnp.zeros((2,))
dummy_xi = jnp.zeros((1,))

actor_params = actor.init(subkeys[0], dummy_z, dummy_y)
critic1_params = critic.init(subkeys[1], dummy_z, dummy_y, dummy_act)
critic2_params = critic.init(subkeys[2], dummy_z, dummy_y, dummy_act)
surrogate_params = surrogate.init(subkeys[3], dummy_y, dummy_act, dummy_xi) 

target_actor_params = actor_params
target_critic1_params = critic1_params
target_critic2_params = critic2_params

# Lowered learning rate for Actor to prevent Tanh saturation
tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-5)) 
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
tx_surrogate = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))

opt_actor = tx_actor.init(actor_params)
opt_critic1 = tx_critic.init(critic1_params)
opt_critic2 = tx_critic.init(critic2_params)
opt_surrogate = tx_surrogate.init(surrogate_params)

# --- 1. DYNAMIC REPLAY BUFFER ---
@struct.dataclass
class DeviceReplayBuffer:
    s: jnp.ndarray
    xi: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    ns: jnp.ndarray
    nxi: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, s_dim, a_dim):
        return cls(
            s=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            xi=jnp.zeros((max_size, N_AGENTS), dtype=jnp.float32),
            a=jnp.zeros((max_size, N_AGENTS, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ns=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            nxi=jnp.zeros((max_size, N_AGENTS), dtype=jnp.float32),
            d=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ptr=jnp.int32(0), size=jnp.int32(0), max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, s_b, xi_b, a_b, r_b, ns_b, nxi_b, d_b):
    batch_size = s_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    new_s = buffer.s.at[indices].set(s_b)
    new_xi = buffer.xi.at[indices].set(xi_b)
    new_a = buffer.a.at[indices].set(a_b)
    new_r = buffer.r.at[indices].set(r_b)
    new_ns = buffer.ns.at[indices].set(ns_b)
    new_nxi = buffer.nxi.at[indices].set(nxi_b)
    new_d = buffer.d.at[indices].set(d_b)
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    return buffer.replace(s=new_s, xi=new_xi, a=new_a, r=new_r, ns=new_ns, nxi=new_nxi, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.s[indices], buffer.xi[indices], buffer.a[indices], buffer.r[indices], buffer.ns[indices], buffer.nxi[indices], buffer.d[indices]

agent_buffer = DeviceReplayBuffer.create(100_000, stored_obs_dim, 2)
rom_buffer = DeviceReplayBuffer.create(100_000, stored_obs_dim, 2)

# --- 2. PURE JAX OBSERVATION BUILDER ---
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
        return extract_patches_jax(state, target, xi, window_size)
    return jax.vmap(single_env_obs)(z_batch, target_batch, xi_batch)

# --- 3. JIT TRAINING & ROLLOUT ---
@jax.jit
def train_surrogate_step(s_params, opt_s, y, u, xi, next_y, next_xi):
    def loss_fn(p):
        vmap_surr = jax.vmap(surrogate.apply, in_axes=(None, 0, 0, 0))
        pred_ny, pred_nxi = vmap_surr(p, y, u, xi)
        loss_y = jnp.mean((pred_ny - next_y)**2)
        loss_xi = jnp.mean((pred_nxi - next_xi)**2)
        return loss_y + loss_xi
    loss, grads = jax.value_and_grad(loss_fn)(s_params)
    updates, opt_s = tx_surrogate.update(grads, opt_s)
    return optax.apply_updates(s_params, updates), opt_s

@jax.jit
def update_critics(c1_p, c2_p, ta_p, tc1_p, tc2_p, opt_c1, opt_c2, y, z, u, r, ny, nz, d, key):
    key, noise_key = jax.random.split(key)
    noise_scale = jnp.array([U_MAX, V_MAX]) * 0.1
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * noise_scale, -0.5 * noise_scale, 0.5 * noise_scale)
    vmap_actor = jax.vmap(actor.apply, in_axes=(None, 0, 0))
    vmap_critic = jax.vmap(critic.apply, in_axes=(None, 0, 0, 0))
    next_u = jnp.clip(vmap_actor(ta_p, nz, ny) + noise, jnp.array([-U_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX]))
    q1_target = jnp.expand_dims(vmap_critic(tc1_p, nz, ny, next_u), -1)
    q2_target = jnp.expand_dims(vmap_critic(tc2_p, nz, ny, next_u), -1)
    min_q = jnp.minimum(q1_target, q2_target)
    q_batch_min, q_batch_max = jnp.min(min_q), jnp.max(min_q)
    clipped_min_q = jnp.clip(min_q, q_batch_min, q_batch_max) 
    target_q = jax.lax.stop_gradient(r + 0.99 * (1.0 - d) * clipped_min_q)
    def q_loss_fn(p):
        q_pred = jnp.expand_dims(vmap_critic(p, z, y, u), -1)
        return jnp.mean(optax.huber_loss(q_pred, target_q))
    loss_c1, grads_c1 = jax.value_and_grad(q_loss_fn)(c1_p)
    loss_c2, grads_c2 = jax.value_and_grad(q_loss_fn)(c2_p)
    up_c1, opt_c1 = tx_critic.update(grads_c1, opt_c1)
    up_c2, opt_c2 = tx_critic.update(grads_c2, opt_c2)
    return optax.apply_updates(c1_p, up_c1), optax.apply_updates(c2_p, up_c2), opt_c1, opt_c2

@jax.jit
def update_actor_and_targets(a_p, c1_p, c2_p, ta_p, tc1_p, tc2_p, opt_a, y, z):
    def a_loss_fn(p):
        vmap_actor = jax.vmap(actor.apply, in_axes=(None, 0, 0))
        vmap_critic = jax.vmap(critic.apply, in_axes=(None, 0, 0, 0))
        act = vmap_actor(p, z, y)
        q1 = vmap_critic(c1_p, z, y, act)
        q2 = vmap_critic(c2_p, z, y, act)
        return -jnp.mean(0.5 * (q1 + q2))
    loss_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    tau = 0.005
    ta_p = jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, a_p, ta_p)
    tc1_p = jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, c1_p, tc1_p)
    tc2_p = jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, c2_p, tc2_p)
    return a_p, ta_p, tc1_p, tc2_p, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, y_batch, xi_batch, key, add_noise=True):
    pe_batch = jax.vmap(lambda xi: get_sinusoidal_encoding(xi, d=pe_dim))(xi_batch)
    z_batch = pe_batch
    vmap2_actor = jax.vmap(jax.vmap(actor.apply, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))
    actions = vmap2_actor(a_p, z_batch, y_batch)
    if add_noise:
        noise_scale = jnp.array([U_MAX, V_MAX]) * 0.1
        noise = jax.random.normal(key, actions.shape) * noise_scale
        actions = jnp.clip(actions + noise, jnp.array([-U_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX]))
    return actions

@jax.jit
def parallel_marl_physics_step(z_batch, xi_batch, target_batch, actions, key):
    keys = jax.random.split(key, z_batch.shape[0])
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, 
            t_steps=1
        )
        return traj[0][-1], traj[1][-1]
    
    next_z_batch, next_xi_batch = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    is_invalid = jnp.logical_not(jnp.isfinite(next_z_batch).all(axis=-1, keepdims=True))
    dones_batch = is_invalid
    safe_z = jnp.where(dones_batch, jnp.zeros_like(next_z_batch), next_z_batch)
    safe_xi = jnp.where(dones_batch, xi_batch, next_xi_batch)
    next_obs_batch_no_pe = build_marl_obs_batch(safe_z, target_batch, safe_xi)
    
    u_batch, v_batch = actions[..., 0], actions[..., 1]
    center_errors = next_obs_batch_no_pe[:, :, 10]
    
    # NORMALIZED REWARD SCALING: Avoids saturation wall
    r_track = -0.1 * jnp.square(center_errors)
    r_effort = -0.0001 * (jnp.square(u_batch) + 0.1 * jnp.square(v_batch))
    r_bound = -10.0 * (jnp.maximum(0.0, 0.02 - safe_xi)**2 + jnp.maximum(0.0, safe_xi - 0.98)**2)
    rewards_batch = (r_track + r_effort + r_bound)[..., None]
    
    return safe_z, safe_xi, next_obs_batch_no_pe, rewards_batch, dones_batch

@jax.jit
def fast_imagination_step(s_params, a_p, obs, xi, key):
    act = get_batch_actions(a_p, obs, xi, key, add_noise=True)
    y_flat, act_flat, xi_flat = obs.reshape(-1, local_y_dim), act.reshape(-1, 2), xi.reshape(-1, 1)
    vmap_surr = jax.vmap(surrogate.apply, in_axes=(None, 0, 0, 0))
    ny_flat, nxi_flat = vmap_surr(s_params, y_flat, act_flat, xi_flat)
    ny, nxi = ny_flat.reshape(obs.shape), jnp.clip(nxi_flat.reshape(xi.shape), 0.0, 1.0)
    
    is_invalid = jnp.logical_not(jnp.isfinite(ny).all(axis=-1, keepdims=True))
    safe_ny = jnp.where(is_invalid, jnp.zeros_like(ny), ny)
    u_batch, v_batch = act[..., 0], act[..., 1]
    
    r_track = -0.1 * jnp.square(safe_ny[:, :, 10])
    r_effort = -0.0001 * (jnp.square(u_batch) + 0.1 * jnp.square(v_batch))
    r_bound = -10.0 * (jnp.maximum(0.0, 0.02 - nxi)**2 + jnp.maximum(0.0, nxi - 0.98)**2)
    rewards = jnp.where(is_invalid, -100.0, (r_track + r_effort + r_bound))[..., None]
    
    return safe_ny, nxi, act, rewards, is_invalid

@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_z, init_xi, target_z, max_steps, key):
    def step_fn(state_tuple, _):
        z_curr, xi_curr, k = state_tuple
        k, subk = jax.random.split(k)
        obs_no_pe = build_marl_obs_batch(z_curr[None, ...], target_z[None, ...], xi_curr[None, ...]) 
        act = get_batch_actions(actor_params, obs_no_pe, xi_curr[None, ...], None, add_noise=False)
        act_flat = act.squeeze(0)
        traj = dynamics.unroll_controlled(
            z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=act_flat, t_steps=1
        )
        next_z, next_xi = traj[0][-1], traj[1][-1]
        mse = jnp.mean((next_z - target_z)**2)
        crashed = jnp.isnan(next_z).any() | jnp.isinf(next_z).any()
        return (next_z, next_xi, k), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi, key), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# --- Training Loop ---
print("Pre-generating banks...")
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.4))(bank_keys)
xi_init_single = jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32)

key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, 1000)
z_batch, target_batch, xi_batch = z_init_bank[idx], z_target_bank[idx], jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))
obs_batch, env_step_counts = build_marl_obs_batch(z_batch, target_batch, xi_batch), jnp.zeros(NUM_PARALLEL_ENVS)

print("Starting Training (Heat)...")
start_time = time.time()
python_agent_buffer_size = 0

for update_step in range(TOTAL_UPDATES):
    if update_step % EVAL_INT == 0:
        key, eval_key = jax.random.split(key)
        eval_e, crashed = fast_eval_episode(actor_params, z_init_bank[0], xi_init_single, z_target_bank[0], MAX_ENV_STEPS, eval_key)
        print(f"Update {update_step:06d} | MSE: {'[CRASHED]' if crashed else f'{eval_e:.6f}'} | Time: {time.time()-start_time:.1f}s")

    key, act_key, physics_key, reset_key = jax.random.split(key, 4)
    actions = jax.random.uniform(act_key, (NUM_PARALLEL_ENVS, N_AGENTS, 2), minval=-1.0, maxval=1.0) if update_step < WARMUP_UPDATES else get_batch_actions(actor_params, obs_batch, xi_batch, act_key)
        
    next_z, next_xi, next_obs, rewards, dones = parallel_marl_physics_step(z_batch, xi_batch, target_batch, actions, physics_key)
    env_step_counts += 1
    truncations = env_step_counts >= MAX_ENV_STEPS
    agent_buffer = add_batch_to_buffer(agent_buffer, obs_batch, xi_batch, actions, rewards, next_obs, next_xi, jnp.tile(dones[:, None, :], (1, N_AGENTS, 1)))
    python_agent_buffer_size = min(python_agent_buffer_size + NUM_PARALLEL_ENVS, 100_000)

    needs_reset = jnp.logical_or(dones.flatten(), truncations)
    idx_reset = jax.random.randint(reset_key, (NUM_PARALLEL_ENVS,), 0, 1000)
    z_batch, target_batch, xi_batch = jnp.where(needs_reset[:, None], z_init_bank[idx_reset], next_z), jnp.where(needs_reset[:, None], z_target_bank[idx_reset], target_batch), jnp.where(needs_reset[:, None], jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1)), next_xi)
    obs_batch, env_step_counts = build_marl_obs_batch(z_batch, target_batch, xi_batch), jnp.where(needs_reset, 0, env_step_counts)
        
    if python_agent_buffer_size > ENV_BATCH_SIZE:
        bx, bxi, bu, br, bnx, bnxi, bd = sample_buffer(agent_buffer, ENV_BATCH_SIZE, subkey) 
        key, subkey = jax.random.split(key)
        bz, bnz = jax.vmap(lambda xi: get_sinusoidal_encoding(xi, d=pe_dim))(bxi), jax.vmap(lambda xi: get_sinusoidal_encoding(xi, d=pe_dim))(bnxi)
        
        critic1_params, critic2_params, opt_critic1, opt_critic2 = update_critics(critic1_params, critic2_params, target_actor_params, target_critic1_params, target_critic2_params, opt_critic1, opt_critic2, bx.reshape(-1, local_y_dim), bz.reshape(-1, z_dim), bu.reshape(-1, 2), br.reshape(-1, 1), bnx.reshape(-1, local_y_dim), bnz.reshape(-1, z_dim), bd.reshape(-1, 1), subkey)
        
        if update_step % POLICY_DELAY == 0:
            actor_params, target_actor_params, target_critic1_params, target_critic2_params, opt_actor = update_actor_and_targets(actor_params, critic1_params, critic2_params, target_actor_params, target_critic1_params, target_critic2_params, opt_actor, bx.reshape(-1, local_y_dim), bz.reshape(-1, z_dim))

# Save
with open('models/hypemarl_heat_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params, 'critic1': critic1_params, 'critic2': critic2_params}))
print("Training finished.")