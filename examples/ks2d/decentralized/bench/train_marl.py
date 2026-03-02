import jax
import jax.numpy as jnp
import optax
import flax.serialization
from flax import struct
import numpy as np
import time
import pickle
from pathlib import Path
import sys
from functools import partial

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from env_ks2d import KS2DMARLEnv, extract_patches_2d_jit
from models_marl import MARLActor2DKS, MARLCritic2DKS, U_MAX
from utils_hypemarl import get_sinusoidal_encoding
from examples.ks2d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks2d.decentralized.dynamics_dual import PDEDynamics2D 

# --- Configurations ---
N_AGENTS = 100
L_DOMAIN = 32.0
N_GRID = 64

ENV_BATCH_SIZE = 128 
EVAL_INT = 10
POLICY_DELAY = 2 

# KS2D Specific Control Timing
MAX_ENV_STEPS = 50     # Control steps (T_steps)
SUBSTEPS = 20          # Physics steps per control step
DT = 0.005             # Physics dt

# Vectorization Configs
NUM_PARALLEL_ENVS = 64
TOTAL_UPDATES = 1000 #100000 
WARMUP_UPDATES = 500

def get_2d_sinusoidal_encoding(p_2d, d=1024, n=1000.0):
    """Combines two 1D positional encodings for 2D coordinates."""
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    # Action shape is (Batch, N_agents), unroll_controlled handles this shape
    return action_params

dynamics = PDEDynamics2D(policy_apply_fn=direct_control_policy)

# Initialize dummy env to extract static parameters
dummy_pool = jnp.zeros((1, N_GRID, N_GRID))
env = KS2DMARLEnv(
    dynamics, initial_conditions=dummy_pool, n_agents=N_AGENTS, 
    N_grid=N_GRID, L=L_DOMAIN, dt=DT, substeps=SUBSTEPS, max_steps=MAX_ENV_STEPS
)

patch_size = env.patch_size # 12
local_y_dim = env.local_y_dim # 3 * 144 = 432
n_mu = env.n_mu # 2 (L, dt)
pe_dim = 2048 # 1024 * 2 (X and Y)

stored_obs_dim = local_y_dim + n_mu 
total_input_dim = stored_obs_dim + pe_dim

# Extract static variables for JAX
xi_fixed = jnp.array(env.agent_positions)
xi_norm = jnp.array(env.xi_norm)
mu_jax = jnp.array(env.mu)
pe_jax = jnp.array(get_2d_sinusoidal_encoding(xi_norm, d=1024))
target_state = jnp.zeros((N_GRID, N_GRID))

actor = MARLActor2DKS()
critic = MARLCritic2DKS()

key, *subkeys = jax.random.split(key, 4)
dummy_input = jnp.zeros((ENV_BATCH_SIZE, total_input_dim))
dummy_u = jnp.zeros((ENV_BATCH_SIZE, 1))

actor_params = actor.init(subkeys[0], dummy_input)
critic_params = critic.init(subkeys[1], dummy_input, dummy_u)

target_actor_params = actor_params
target_critic_params = critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. ON-DEVICE REPLAY BUFFER (3D) ---
@struct.dataclass
class DeviceReplayBuffer:
    s: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    ns: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, s_dim, a_dim):
        return cls(
            s=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            a=jnp.zeros((max_size, N_AGENTS, a_dim), dtype=jnp.float32),
            r=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ns=jnp.zeros((max_size, N_AGENTS, s_dim), dtype=jnp.float32),
            d=jnp.zeros((max_size, N_AGENTS, 1), dtype=jnp.float32),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, s_batch, a_batch, r_batch, ns_batch, d_batch):
    batch_size = s_batch.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_s = buffer.s.at[indices].set(s_batch)
    new_a = buffer.a.at[indices].set(a_batch)
    new_r = buffer.r.at[indices].set(r_batch)
    new_ns = buffer.ns.at[indices].set(ns_batch)
    new_d = buffer.d.at[indices].set(d_batch)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(s=new_s, a=new_a, r=new_r, ns=new_ns, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.s[indices], buffer.a[indices], buffer.r[indices], buffer.ns[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(12_500, stored_obs_dim, 1)

# --- 2. PURE JAX OBSERVATION BUILDER ---
@jax.jit
def build_marl_obs_batch(full_state_batch):
    def single_env_obs(state):
        y_local = extract_patches_2d_jit(state, target_state, xi_norm, patch_size, N_GRID)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return jnp.concatenate([y_local, mu_broadcast], axis=-1)
    return jax.vmap(single_env_obs)(full_state_batch)

# --- 3. JIT TRAINING & ROLLOUT ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, x, u, r, nx, d, key):
    key, noise_key = jax.random.split(key)
    
    # Scale proportional noise dynamically using U_MAX
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * (U_MAX * 0.1), -U_MAX * 0.5, U_MAX * 0.5)
    next_u = jnp.clip(actor.apply(ta_p, nx) + noise, -U_MAX, U_MAX)
    
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
def get_batch_actions(a_p, obs_batch_no_pe, key, add_noise=True):
    pe_expanded = jnp.tile(pe_jax[None, :, :], (obs_batch_no_pe.shape[0], 1, 1))
    full_obs = jnp.concatenate([obs_batch_no_pe, pe_expanded], axis=-1)
    
    actions = jax.vmap(jax.vmap(actor.apply, in_axes=(None, 0)), in_axes=(None, 0))(a_p, full_obs)
    
    if add_noise:
        noise = jax.random.normal(key, actions.shape) * (U_MAX * 0.1)
        actions = jnp.clip(actions + noise, -U_MAX, U_MAX)
    return actions

@jax.jit
def parallel_marl_physics_step(u_batch, actions):
    acts_flat = actions.squeeze(-1) 
    
    def single_physics_step(u_single, act_single):
        traj = dynamics.unroll_controlled(
            u_init=u_single, xi_fixed=xi_fixed, u_target=target_state, params=act_single, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        return traj[0][-1]
    
    next_u_batch = jax.vmap(single_physics_step)(u_batch, acts_flat)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_u_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_u_batch), axis=(1, 2)) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_u = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_u_batch), next_u_batch)
    next_obs_batch_no_pe = build_marl_obs_batch(safe_u)
    
    # 2D KS Reward Logic
    # 1. Global Energy
    global_energy = jnp.mean(jnp.square(safe_u), axis=(1, 2))
    
    # 2. Local Stability (First 144 elements correspond to the `error` patch)
    y_local_err = next_obs_batch_no_pe[..., :patch_size**2] 
    local_rewards = -jnp.mean(jnp.square(y_local_err), axis=-1)
    
    # Combine local and global rewards exactly as in KS2DMARLEnv
    rewards_batch = 0.5 * local_rewards + 0.5 * (-global_energy[:, None])
    rewards_batch = rewards_batch[..., None] 
    
    return safe_u, next_obs_batch_no_pe, rewards_batch, dones_batch

# --- 4. FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, max_steps):
    def step_fn(state, _):
        obs_no_pe = build_marl_obs_batch(state[None, ...]) 
        act = get_batch_actions(actor_params, obs_no_pe, None, add_noise=False)
        act_flat = act.squeeze() 
        
        traj = dynamics.unroll_controlled(
            u_init=state, xi_fixed=xi_fixed, u_target=target_state, params=act_flat, 
            t_steps=1, substeps=SUBSTEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=1.2
        )
        next_state = traj[0][-1]
        
        energy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 100.0)
        
        return next_state, (energy, crashed)

    _, (energies, crashes) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    return jnp.mean(energies), jnp.any(crashes)

# --- Vectorized Training Loop ---
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
obs_batch = build_marl_obs_batch(u_batch)
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

python_buffer_size = 0

print("Starting Massively Parallel MARL Training (2D KS)...")
start_time = time.time()

for update_step in range(TOTAL_UPDATES):
    
    if update_step % EVAL_INT == 0:
        eval_u = state_bank[0] 
        eval_e, crashed = fast_eval_episode(actor_params, eval_u, MAX_ENV_STEPS)
        episode_num = update_step // MAX_ENV_STEPS
        
        if crashed:
            print(f"Update {update_step:06d} | Episode {episode_num} | Eval Energy: [CRASHED] | Time: {time.time()-start_time:.1f}s")
        else:
            print(f"Update {update_step:06d} | Episode {episode_num} | Eval Energy: {eval_e:.6f} | Time: {time.time()-start_time:.1f}s")

    # 2. Parallel Data Collection 
    key, act_key, physics_key, reset_key = jax.random.split(key, 4)
    
    if update_step < WARMUP_UPDATES:
        actions = jax.random.uniform(act_key, (NUM_PARALLEL_ENVS, N_AGENTS, 1), minval=-U_MAX, maxval=U_MAX)
    else:
        actions = get_batch_actions(actor_params, obs_batch, act_key, add_noise=True)
        
    next_u_batch, next_obs_batch, rewards_batch, dones_batch = parallel_marl_physics_step(u_batch, actions)
    
    env_step_counts += 1
    truncations_batch = env_step_counts >= MAX_ENV_STEPS
    
    safe_rewards = jnp.where(dones_batch[:, None, None], -100.0, rewards_batch)
    dones_expanded = jnp.tile(dones_batch[:, None, None], (1, N_AGENTS, 1))

    buffer = add_batch_to_buffer(buffer, obs_batch, actions, safe_rewards, next_obs_batch, dones_expanded)
    
    # 4. Handle Resets 
    needs_reset = jnp.logical_or(dones_batch.flatten(), truncations_batch)
    fresh_states = jax.random.choice(reset_key, state_bank, shape=(NUM_PARALLEL_ENVS,))
    
    # Needs_reset handles broadcasting correctly to replace 2D fields
    u_batch = jnp.where(needs_reset[:, None, None], fresh_states, next_u_batch)
    
    obs_batch = build_marl_obs_batch(u_batch)
    env_step_counts = jnp.where(needs_reset, 0, env_step_counts)
        
    # 5. TD3 Updates
    python_buffer_size = min(python_buffer_size + NUM_PARALLEL_ENVS, 12_500)
    
    if python_buffer_size > ENV_BATCH_SIZE:
        bx, bu, br, bnx, bd = sample_buffer(buffer, ENV_BATCH_SIZE, subkey) 
        key, subkey = jax.random.split(key)
        
        bx_flat = bx.reshape(-1, stored_obs_dim)
        bu_flat = bu.reshape(-1, 1)
        br_flat = br.reshape(-1, 1)
        bnx_flat = bnx.reshape(-1, stored_obs_dim)
        bd_flat = bd.reshape(-1, 1)
        
        pe_tiled = jnp.tile(pe_jax, (ENV_BATCH_SIZE, 1))
        
        bx_full = jnp.concatenate([bx_flat, pe_tiled], axis=-1)
        bnx_full = jnp.concatenate([bnx_flat, pe_tiled], axis=-1)
        
        critic_params, opt_critic = update_critic(
            critic_params, target_actor_params, target_critic_params, opt_critic, bx_full, bu_flat, br_flat, bnx_full, bd_flat, subkey
        )
        
        if update_step % POLICY_DELAY == 0:
            actor_params, target_actor_params, target_critic_params, opt_actor = update_actor_and_targets(
                actor_params, critic_params, target_actor_params, target_critic_params, opt_actor, bx_full
            )

# Save
with open('models/marl_ks2d_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")