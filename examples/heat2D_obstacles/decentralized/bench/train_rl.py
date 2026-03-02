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

# Project imports for 2D Heat Equation
from models_rl import CentralizedActor2D, CentralizedCritic2D, U_MAX, V_MAX
from examples.heat2D.decentralized.data_utils import get_training_data
from examples.heat2D.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 16  
L_DOMAIN = 1.0
N_GRID = 32
BATCH_SIZE = 256
EVAL_INT = 500
POLICY_DELAY = 2
MAX_ENV_STEPS = 100 

# Vectorization Configs
NUM_PARALLEL_ENVS = 128
TOTAL_UPDATES = 5000 # 100000
WARMUP_UPDATES = 500

# --- Obstacles Config ---
OBSTACLES = jnp.array([
    [0.30, 0.30, 0.06],   # x_center, y_center, radius
    [0.50, 0.50, 0.06],
    [0.70, 0.70, 0.06],
])
R_SAFE_OBSTACLE = 0.04
# -----------------------------

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    # action_params has shape (N_AGENTS, 3)
    u = action_params[:, 0]
    v = action_params[:, 1:3]
    return u, v

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)

actor = CentralizedActor2D(n_agents=N_AGENTS)
critic = CentralizedCritic2D()

key, *subkeys = jax.random.split(key, 4)
dummy_z = jnp.zeros((BATCH_SIZE, N_GRID, N_GRID))
dummy_target = jnp.zeros((BATCH_SIZE, N_GRID, N_GRID))
dummy_xi = jnp.zeros((BATCH_SIZE, N_AGENTS, 2))
dummy_action = jnp.zeros((BATCH_SIZE, N_AGENTS, 3))

actor_params = actor.init(subkeys[0], dummy_z, dummy_target, dummy_xi)
critic_params = critic.init(subkeys[1], dummy_z, dummy_target, dummy_xi, dummy_action)
target_actor_params, target_critic_params = actor_params, critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. ON-DEVICE REPLAY BUFFER ---
@struct.dataclass
class DeviceReplayBuffer:
    z: jnp.ndarray
    zt: jnp.ndarray
    xi: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    nz: jnp.ndarray
    nxi: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, n_grid, n_agents, a_dim):
        return cls(
            z=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float32),
            zt=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float32),
            xi=jnp.zeros((max_size, n_agents, 2), dtype=jnp.float32),
            a=jnp.zeros((max_size, n_agents, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, 1), dtype=jnp.float32),
            nz=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float32),
            nxi=jnp.zeros((max_size, n_agents, 2), dtype=jnp.float32),
            d=jnp.zeros((max_size, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, z_b, zt_b, xi_b, a_b, r_b, nz_b, nxi_b, d_b):
    batch_size = z_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_z = buffer.z.at[indices].set(z_b)
    new_zt = buffer.zt.at[indices].set(zt_b)
    new_xi = buffer.xi.at[indices].set(xi_b)
    new_a = buffer.a.at[indices].set(a_b)
    new_r = buffer.r.at[indices].set(r_b)
    new_nz = buffer.nz.at[indices].set(nz_b)
    new_nxi = buffer.nxi.at[indices].set(nxi_b)
    new_d = buffer.d.at[indices].set(d_b)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(z=new_z, zt=new_zt, xi=new_xi, a=new_a, r=new_r, nz=new_nz, nxi=new_nxi, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.z[indices], buffer.zt[indices], buffer.xi[indices], buffer.a[indices], buffer.r[indices], buffer.nz[indices], buffer.nxi[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(125_000, N_GRID, N_AGENTS, 3)

# --- JIT TRAINING & ROLLOUT FUNCTIONS ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, z, zt, xi, a, r, nz, nxi, d, key): 
    key, noise_key = jax.random.split(key)
    
    noise_scale = jnp.array([U_MAX, V_MAX, V_MAX]) * 0.1
    noise = jnp.clip(jax.random.normal(noise_key, a.shape) * noise_scale, -0.5 * noise_scale, 0.5 * noise_scale)
    
    raw_next_a = actor.apply(ta_p, nz, zt, nxi) + noise
    next_a = jnp.clip(raw_next_a, jnp.array([-U_MAX, -V_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX, V_MAX]))
    
    t_q1, t_q2 = critic.apply(tc_p, nz, zt, nxi, next_a)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, z, zt, xi, a)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, z, zt, xi):
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, z, zt, xi, actor.apply(p, z, zt, xi))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, z_batch, z_target_batch, xi_batch, key, add_noise=True):
    actions = jax.vmap(actor.apply, in_axes=(None, 0, 0, 0))(a_p, z_batch, z_target_batch, xi_batch)
    if add_noise:
        noise_scale = jnp.array([U_MAX, V_MAX, V_MAX]) * 0.1
        noise = jax.random.normal(key, actions.shape) * noise_scale
        actions = jnp.clip(actions + noise, jnp.array([-U_MAX, -V_MAX, -V_MAX]), jnp.array([U_MAX, V_MAX, V_MAX]))
    return actions

@jax.jit
def parallel_physics_step(z_batch, xi_batch, target_batch, actions, prev_v_batch, key):
    keys = jax.random.split(key, z_batch.shape[0])
    
    def single_physics_step(z_s, xi_s, target_s, act_s, k_s):
        traj = dynamics.unroll_controlled(
            z_init=z_s, xi_init=xi_s, z_target=target_s, params=act_s, 
            t_steps=1
        )
        return traj[0][-1], traj[1][-1]
    
    next_z_batch, next_xi_batch = jax.vmap(single_physics_step)(z_batch, xi_batch, target_batch, actions, keys)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_z_batch).all(axis=(1, 2)))
    dones_batch = is_invalid[:, None] # Keep dims for where operations
    
    safe_z = jnp.where(dones_batch[:, :, None], jnp.zeros_like(next_z_batch), next_z_batch)
    safe_xi = jnp.where(dones_batch[:, :, None], xi_batch, next_xi_batch)
    
    u_batch = actions[..., 0]
    v_batch = actions[..., 1:3]
    
    # --- Formulate Global Reward safely shaped as (BATCH, 1) ---
    
    mse = jnp.mean((safe_z - target_batch)**2, axis=(1, 2))[:, None]
    effort = jnp.mean(0.001 * (jnp.square(u_batch) + 0.1 * jnp.sum(jnp.square(v_batch), axis=-1)), axis=-1)[:, None]
    
    margin = 0.02
    x_pen = jnp.maximum(0.0, margin - safe_xi[..., 0])**2 + jnp.maximum(0.0, safe_xi[..., 0] - (1.0 - margin))**2
    y_pen = jnp.maximum(0.0, margin - safe_xi[..., 1])**2 + jnp.maximum(0.0, safe_xi[..., 1] - (1.0 - margin))**2
    mean_oob_penalty = jnp.mean(100.0 * (x_pen + y_pen), axis=-1)[:, None]
    
    # Agent-Agent Collision Penalty
    R_safe = 0.08
    diff = safe_xi[:, :, None, :] - safe_xi[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    mask = jnp.eye(N_AGENTS)[None, :, :]
    coll_agents = 20.0 * jnp.sum(jnp.maximum(0.0, R_safe - (dists + mask * 10.0)) ** 2, axis=2)
    mean_coll_agents = jnp.mean(coll_agents, axis=-1)[:, None]
    
    # --- NEW: Agent-Obstacle Collision Penalty ---
    obstacle_centers = OBSTACLES[:, :2]
    obstacle_radii = OBSTACLES[:, 2]
    
    diff_obs = safe_xi[:, :, None, :] - obstacle_centers[None, None, :, :]
    dists_obs = jnp.sqrt(jnp.sum(diff_obs**2, axis=-1) + 1e-8)
    
    safety_dist = R_SAFE_OBSTACLE + obstacle_radii[None, None, :]
    coll_obstacles = 100.0 * jnp.sum(jnp.maximum(0.0, safety_dist - dists_obs)**2, axis=2)
    mean_coll_obstacles = jnp.mean(coll_obstacles, axis=-1)[:, None]
    # ---------------------------------------------

    mean_accel_penalty = jnp.mean(0.1 * jnp.sum(jnp.square(v_batch - prev_v_batch), axis=-1), axis=-1)[:, None]
    
    # Combine into a unified reward signal (All strictly shaped Batch x 1)
    rewards_batch = -mse - effort - mean_oob_penalty - mean_coll_agents - mean_coll_obstacles - mean_accel_penalty
    
    return safe_z, safe_xi, rewards_batch, dones_batch, v_batch

# --- 2. FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_z, init_xi, target_z, max_steps, key):
    def step_fn(state_tuple, _):
        z_curr, xi_curr, k = state_tuple
        k, subk = jax.random.split(k)
        
        act = actor.apply(actor_params, z_curr[None, ...], target_z[None, ...], xi_curr[None, ...])
        act_flat = act.squeeze(0)
        
        traj = dynamics.unroll_controlled(
            z_init=z_curr, xi_init=xi_curr, z_target=target_z, params=act_flat, 
            t_steps=1
        )
        next_z, next_xi = traj[0][-1], traj[1][-1]
        
        mse = jnp.mean((next_z - target_z)**2)
        crashed = jnp.isnan(next_z).any() | jnp.isinf(next_z).any()
        
        return (next_z, next_xi, k), (mse, crashed)

    _, (mses, crashes) = jax.lax.scan(step_fn, (init_z, init_xi, key), None, length=max_steps)
    return jnp.mean(mses), jnp.any(crashes)

# --- Vectorized Training Loop ---
print("Loading 2D starting state & target banks from dataset...")
z_init_all, z_target_all, _ = get_training_data(n_samples=5000, n_grid=N_GRID, dataset_dir='../../data')
z_init_bank = jnp.array(z_init_all)
z_target_bank = jnp.array(z_target_all)

# 4x4 Agent Template
n_side = int(np.sqrt(N_AGENTS))
pos_1d = np.linspace(0.2, 0.8, n_side)
X, Y = np.meshgrid(pos_1d, pos_1d)
xi_init_single = jnp.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)

key, subkey = jax.random.split(key)
idx = jax.random.randint(subkey, (NUM_PARALLEL_ENVS,), 0, len(z_init_bank))
z_batch = z_init_bank[idx]
target_batch = z_target_bank[idx]
xi_batch = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1, 1))
prev_v_batch = jnp.zeros((NUM_PARALLEL_ENVS, N_AGENTS, 2))

env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)
python_buffer_size = 0

print("Starting Massively Parallel Centralized RL Training (2D Heat Equation with Obstacles)...")
start_time = time.time()

for update_step in range(TOTAL_UPDATES):
    
    if update_step % EVAL_INT == 0:
        eval_z, eval_target = z_init_bank[0], z_target_bank[0]
        eval_xi = xi_init_single
        key, eval_key = jax.random.split(key)
        
        eval_e, crashed = fast_eval_episode(actor_params, eval_z, eval_xi, eval_target, MAX_ENV_STEPS, eval_key)
        episode_num = update_step // MAX_ENV_STEPS
        
        if crashed:
            print(f"Update {update_step:06d} | Episode {episode_num} | Eval Tracking MSE: [CRASHED] | Time: {time.time()-start_time:.1f}s")
        else:
            print(f"Update {update_step:06d} | Episode {episode_num} | Eval Tracking MSE: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

    # 2. Parallel Data Collection 
    key, act_key, physics_key, reset_key = jax.random.split(key, 4)
    
    if update_step < WARMUP_UPDATES:
        actions = jax.random.uniform(act_key, (NUM_PARALLEL_ENVS, N_AGENTS, 3), 
                                     minval=jnp.array([-U_MAX, -V_MAX, -V_MAX]), 
                                     maxval=jnp.array([U_MAX, V_MAX, V_MAX]))
    else:
        actions = get_batch_actions(actor_params, z_batch, target_batch, xi_batch, act_key, add_noise=True)
        
    next_z_batch, next_xi_batch, rewards_batch, dones_batch, prev_v_batch = parallel_physics_step(
        z_batch, xi_batch, target_batch, actions, prev_v_batch, physics_key
    )
    
    env_step_counts += 1
    truncations_batch = env_step_counts >= MAX_ENV_STEPS
    
    safe_next_z = jnp.where(dones_batch[:, :, None], jnp.zeros_like(next_z_batch), next_z_batch)
    safe_next_xi = jnp.where(dones_batch[:, :, None], xi_batch, next_xi_batch)
    safe_rewards = jnp.where(dones_batch, -100.0, rewards_batch)

    buffer = add_batch_to_buffer(buffer, z_batch, target_batch, xi_batch, actions, safe_rewards, safe_next_z, safe_next_xi, dones_batch)
    
    # 4. Handle Resets 
    needs_reset = jnp.logical_or(dones_batch.flatten(), truncations_batch)
    idx_reset = jax.random.randint(reset_key, (NUM_PARALLEL_ENVS,), 0, len(z_init_bank))
    
    fresh_z = z_init_bank[idx_reset]
    fresh_target = z_target_bank[idx_reset]
    fresh_xi = jnp.tile(xi_init_single, (NUM_PARALLEL_ENVS, 1, 1))
    fresh_v = jnp.zeros((NUM_PARALLEL_ENVS, N_AGENTS, 2))
    
    z_batch = jnp.where(needs_reset[:, None, None], fresh_z, safe_next_z)
    target_batch = jnp.where(needs_reset[:, None, None], fresh_target, target_batch)
    xi_batch = jnp.where(needs_reset[:, None, None], fresh_xi, safe_next_xi)
    prev_v_batch = jnp.where(needs_reset[:, None, None], fresh_v, prev_v_batch)
    
    env_step_counts = jnp.where(needs_reset, 0, env_step_counts)
        
    # 5. TD3 Updates
    python_buffer_size = min(python_buffer_size + NUM_PARALLEL_ENVS, 125_000)
    
    if python_buffer_size > BATCH_SIZE:
        bz, bzt, bxi, ba, br, bnz, bnxi, bd = sample_buffer(buffer, BATCH_SIZE, subkey) 
        key, subkey = jax.random.split(key)
        
        critic_params, opt_critic = update_critic(
            critic_params, target_actor_params, target_critic_params, opt_critic, bz, bzt, bxi, ba, br, bnz, bnxi, bd, subkey
        )
        
        if update_step % POLICY_DELAY == 0:
            actor_params, target_actor_params, target_critic_params, opt_actor = update_actor_and_targets(
                actor_params, critic_params, target_actor_params, target_critic_params, opt_actor, bz, bzt, bxi
            )

# Save
with open('models/rl_heat2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")