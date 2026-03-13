import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import flax.serialization
from flax import struct
import numpy as np
import time
from pathlib import Path
import sys
from functools import partial
from tqdm import trange

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports for Heat Equation
from env_he import HeatHypeMARLEnv 
from utils_hypemarl import get_sinusoidal_encoding
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics 
from models_marl import MARLActor, MARLCritic, U_MAX, V_MAX

# --- Configurations ---
N_AGENTS = 8 
L_DOMAIN = 1.0
N_GRID = 100

ENV_BATCH_SIZE = 256 
EVAL_INT = 500
POLICY_DELAY = 2 
MAX_ENV_STEPS = 300

# Vectorization Configs
NUM_PARALLEL_ENVS = 256
TOTAL_UPDATES = 100000 
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
n_mu = env.n_mu
pe_dim = 128

stored_obs_dim = local_y_dim + n_mu 
total_input_dim = stored_obs_dim + pe_dim

mu_jax = jnp.array(env.mu)
window_size = env.window_size

actor = MARLActor()
critic = MARLCritic()

key, *subkeys = jax.random.split(key, 4)
dummy_input = jnp.zeros((ENV_BATCH_SIZE, total_input_dim))
dummy_act = jnp.zeros((ENV_BATCH_SIZE, 2)) 

actor_params = actor.init(subkeys[0], dummy_input)
critic_params = critic.init(subkeys[1], dummy_input, dummy_act)

target_actor_params = actor_params
target_critic_params = critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4)) 
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

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
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
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
        y_local = extract_patches_jax(state, target, xi, window_size)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return jnp.concatenate([y_local, mu_broadcast], axis=-1)
    return jax.vmap(single_env_obs)(z_batch, target_batch, xi_batch)

# --- 3. JIT TRAINING COMPONENTS ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, x, u, r, nx, d, key):
    key, noise_key = jax.random.split(key)
    noise_scale = jnp.array([U_MAX, V_MAX]) * 0.1
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * noise_scale, -0.5 * noise_scale, 0.5 * noise_scale)
    
    raw_next_u = actor.apply(ta_p, nx) + noise
    next_u = jnp.clip(raw_next_u, jnp.array([-U_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX]))
    
    t_q1, t_q2 = critic.apply(tc_p, nx, next_u)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, x, u)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, x):
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, x, actor.apply(p, x))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, obs_batch_no_pe, xi_batch, key, add_noise=True):
    pe_batch = jax.vmap(lambda xi: get_sinusoidal_encoding(xi, d=pe_dim))(xi_batch)
    full_obs = jnp.concatenate([obs_batch_no_pe, pe_batch], axis=-1)
    
    actions = jax.vmap(jax.vmap(actor.apply, in_axes=(None, 0)), in_axes=(None, 0))(a_p, full_obs)
    
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
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, t_steps=1
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
    
    rewards_batch = (r_track + r_effort + r_bound + r_coll)[..., None]
    
    return safe_z, safe_xi, next_obs_batch_no_pe, rewards_batch, dones_batch

# --- 4. FAST EVALUATION ---
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

# --- 5. THE SCAN-COMPILED TRAINING CHUNK ---
@jax.jit
def train_chunk(carry, step_indices, z_init_bank, z_target_bank):
    """
    Executes a chunk of training steps entirely on the GPU.
    """
    def scan_step(carry, step_idx):
        buf, a_p, c_p, ta_p, tc_p, o_a, o_c, z, target, xi, obs, steps, rng = carry
        rng, act_k, phys_k, res_k, samp_k, net_k = jax.random.split(rng, 6)
        
        # 1. Action Selection (Warmup vs Policy)
        def warmup_actions(_):
            return jax.random.uniform(act_k, (NUM_PARALLEL_ENVS, N_AGENTS, 2), 
                                      minval=jnp.array([-U_MAX, -V_MAX]), maxval=jnp.array([U_MAX, V_MAX]))
        def policy_actions(_):
            return get_batch_actions(a_p, obs, xi, act_k, add_noise=True)
            
        actions = jax.lax.cond(step_idx < WARMUP_UPDATES, warmup_actions, policy_actions, None)
        
        # 2. Physics Step
        nz, nxi, nobs, rew, dones = parallel_marl_physics_step(z, xi, target, actions, phys_k)
        steps += 1
        truncs = steps >= MAX_ENV_STEPS
        needs_reset = jnp.logical_or(dones.flatten(), truncs)
        
        # 3. Update Buffer
        safe_rew = jnp.where(dones[:, None], -100.0, rew)
        dones_exp = jnp.tile(dones[:, None, :], (1, N_AGENTS, 1))
        new_buf = add_batch_to_buffer(buf, obs, xi, actions, safe_rew, nobs, nxi, dones_exp)
        
        # 4. Handle Resets
        idx_reset = jax.random.randint(res_k, (NUM_PARALLEL_ENVS,), 0, 1000)
        fresh_z = z_init_bank[idx_reset]
        fresh_target = z_target_bank[idx_reset]
        fresh_xi = jnp.tile(jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32), (NUM_PARALLEL_ENVS, 1))
        
        z_next = jnp.where(needs_reset[:, None], fresh_z, nz)
        target_next = jnp.where(needs_reset[:, None], fresh_target, target)
        xi_next = jnp.where(needs_reset[:, None], fresh_xi, nxi)
        obs_next = build_marl_obs_batch(z_next, target_next, xi_next)
        steps_next = jnp.where(needs_reset, 0, steps)

        # 5. Network Updates (Conditional on Buffer Size)
        def do_network_updates(net_state):
            c_p, a_p, ta_p, tc_p, o_c, o_a = net_state
            
            bx, bxi, bu, br, bnx, bnxi, bd = sample_buffer(new_buf, ENV_BATCH_SIZE, samp_k)
            bpe = jax.vmap(lambda x_i: get_sinusoidal_encoding(x_i, d=pe_dim))(bxi)
            bnpe = jax.vmap(lambda x_i: get_sinusoidal_encoding(x_i, d=pe_dim))(bnxi)
            
            bx_f = jnp.concatenate([bx.reshape(-1, stored_obs_dim), bpe.reshape(-1, pe_dim)], axis=-1)
            bnx_f = jnp.concatenate([bnx.reshape(-1, stored_obs_dim), bnpe.reshape(-1, pe_dim)], axis=-1)
            bu_f, br_f, bd_f = bu.reshape(-1, 2), br.reshape(-1, 1), bd.reshape(-1, 1)
            
            # Critic Update
            new_c_p, new_o_c = update_critic(c_p, ta_p, tc_p, o_c, bx_f, bu_f, br_f, bnx_f, bd_f, net_k)
            
            # Policy Delayed Actor Update
            def do_actor_update(_):
                return update_actor_and_targets(a_p, new_c_p, ta_p, tc_p, o_a, bx_f)
            def skip_actor_update(_):
                return a_p, ta_p, tc_p, o_a
                
            new_a_p, new_ta_p, new_tc_p, new_o_a = jax.lax.cond(
                step_idx % POLICY_DELAY == 0, do_actor_update, skip_actor_update, None
            )
            
            return new_c_p, new_a_p, new_ta_p, new_tc_p, new_o_c, new_o_a

        def skip_network_updates(net_state):
            return net_state

        net_state = (c_p, a_p, ta_p, tc_p, o_c, o_a)
        c_p, a_p, ta_p, tc_p, o_c, o_a = jax.lax.cond(
            new_buf.size >= ENV_BATCH_SIZE, do_network_updates, skip_network_updates, net_state
        )

        new_carry = (new_buf, a_p, c_p, ta_p, tc_p, o_a, o_c, z_next, target_next, xi_next, obs_next, steps_next, rng)
        return new_carry, None

    return jax.lax.scan(scan_step, carry, step_indices)

# --- 6. MAIN EXECUTION LOOP ---
print("Pre-generating starting state & target banks (Vectorized)...")
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.4))(bank_keys)
xi_init_single = jnp.linspace(0.2, 0.8, N_AGENTS, dtype=jnp.float32)

key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, 1000)
z_init = z_init_bank[idx]
target_init = z_target_bank[idx]
xi_init = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1))
obs_init = build_marl_obs_batch(z_init, target_init, xi_init)
steps_init = jnp.zeros(NUM_PARALLEL_ENVS)
buffer = DeviceReplayBuffer.create(125_000, stored_obs_dim, 2)

# Pack everything into the initial carry state
carry = (
    buffer, actor_params, critic_params, target_actor_params, target_critic_params,
    opt_actor, opt_critic, z_init, target_init, xi_init, obs_init, steps_init, key
)

print(f"Starting Massively Parallel MARL Training (Chunked & JITed)...")
start_time = time.time()

num_chunks = TOTAL_UPDATES // EVAL_INT

for chunk_idx in trange(num_chunks):
    start_step = chunk_idx * EVAL_INT
    step_indices = jnp.arange(start_step, start_step + EVAL_INT)
    
    # Run the compiled chunk
    carry, _ = train_chunk(carry, step_indices, z_init_bank, z_target_bank)
    
    # Unpack the current actor for evaluation
    current_actor_params = carry[1] 
    
    # Evaluation Logic
    eval_z = z_init_bank[0] 
    eval_target = z_target_bank[0]
    key, eval_key = jax.random.split(key)
    
    eval_e, crashed = fast_eval_episode(current_actor_params, eval_z, xi_init_single, eval_target, MAX_ENV_STEPS, eval_key)
    
    current_total_step = start_step + EVAL_INT
    episode_num = current_total_step // MAX_ENV_STEPS
    
    if crashed:
        print(f"Update {current_total_step:06d} | Episode {episode_num} | Eval Tracking MSE: [CRASHED] | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"Update {current_total_step:06d} | Episode {episode_num} | Eval Tracking MSE: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

# Extract final weights and save
final_actor_params = carry[1]
with open('models/marl_heat_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': final_actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")