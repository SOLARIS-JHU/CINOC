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

# Enable x64 for Spectral Stability (crucial for turbulence)
jax.config.update("jax_enable_x64", True)

# Project imports
from env_turb import Turb2DMARLEnv, extract_patches_turb_jit
from models_marl import MARLActor2DTurb, MARLCritic2DTurb, U_MAX
from utils_hypemarl import get_sinusoidal_encoding
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions
from examples.turbulence2d.decentralized.dynamics_dual import PDEDynamics2D 

# --- Configurations ---
N_AGENTS = 64 
L_DOMAIN = 1.0
N_GRID = 64

ENV_BATCH_SIZE = 128 
EVAL_INT = 10
POLICY_DELAY = 2 

MAX_ENV_STEPS = 150    
SUBSTEPS = 5           
DT = 0.01              
VISCOSITY = 5e-4       

NUM_PARALLEL_ENVS = 16 
TOTAL_UPDATES = 1000# 100000 
WARMUP_UPDATES = 50

def get_2d_sinusoidal_encoding(p_2d, d=1024, n=1000.0):
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, xi_fixed, u_obs):
    return action_params

dynamics = PDEDynamics2D(policy_apply_fn=direct_control_policy)

dummy_pool = jnp.zeros((1, N_GRID, N_GRID), dtype=jnp.complex128)
env = Turb2DMARLEnv(dynamics, dummy_pool, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, dt=DT, substeps=SUBSTEPS, max_steps=MAX_ENV_STEPS)

patch_size = env.patch_size 
local_y_dim = env.local_y_dim 
n_mu = env.n_mu 
pe_dim = 2048 

stored_obs_dim = local_y_dim + n_mu 
total_input_dim = stored_obs_dim + pe_dim

xi_fixed = jnp.array(env.agent_positions)
xi_norm = jnp.array(env.xi_norm)
mu_jax = jnp.array(env.mu)
pe_jax = jnp.array(get_2d_sinusoidal_encoding(xi_norm, d=1024))

actor = MARLActor2DTurb()
critic = MARLCritic2DTurb()

key, *subkeys = jax.random.split(key, 4)
dummy_input = jnp.zeros((ENV_BATCH_SIZE, total_input_dim))
dummy_u = jnp.zeros((ENV_BATCH_SIZE, 1))

actor_params = actor.init(subkeys[0], dummy_input)
critic_params = critic.init(subkeys[1], dummy_input, dummy_u)

target_actor_params = actor_params
target_critic_params = critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
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
            ptr=jnp.int32(0), size=jnp.int32(0), max_size=max_size
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

buffer = DeviceReplayBuffer.create(10_000, stored_obs_dim, 1)

# --- 2. PURE JAX OBSERVATION BUILDER ---
@jax.jit
def build_marl_obs_batch(w_hat_batch):
    def single_env_obs(w_hat):
        w_phys = jnp.fft.ifft2(w_hat).real
        y_local = extract_patches_turb_jit(w_phys, xi_norm, patch_size, N_GRID)
        mu_broadcast = jnp.tile(mu_jax, (N_AGENTS, 1))
        return jnp.concatenate([y_local, mu_broadcast], axis=-1)
    return jax.vmap(single_env_obs)(w_hat_batch)

# --- 3. JIT TRAINING & ROLLOUT ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, x, u, r, nx, d, key):
    key, noise_key = jax.random.split(key)
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
def parallel_marl_physics_step(w_hat_batch, actions):
    acts_flat = actions.squeeze(-1) 
    
    def single_physics_step(w_single, act_single):
        w_phys_traj, _ = dynamics.unroll_controlled(
            w_hat_init=w_single, xi_fixed=xi_fixed, params=act_single, 
            t_steps=1, N_grid=N_GRID, L=L_DOMAIN, dt=DT, 
            substeps=SUBSTEPS, viscosity=VISCOSITY, actuator_grid_shape=(8, 8)
        )
        w_phys_next = w_phys_traj[-1]
        w_hat_next = jnp.fft.fft2(w_phys_next)
        return w_hat_next, w_phys_next
    
    next_w_hat_batch, next_w_phys_batch = jax.vmap(single_physics_step)(w_hat_batch, acts_flat)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_w_phys_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_w_phys_batch), axis=(1, 2)) > 100.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_w_hat = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_w_hat_batch), next_w_hat_batch)
    safe_w_phys = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_w_phys_batch), next_w_phys_batch)
    
    next_obs_batch_no_pe = build_marl_obs_batch(safe_w_hat)
    
    # --- REWARD CALCULATION ---
    global_enstrophy = jnp.mean(jnp.square(safe_w_phys), axis=(1, 2))
    
    y_local_w = next_obs_batch_no_pe[..., :patch_size**2] 
    local_rewards = -jnp.mean(jnp.square(y_local_w), axis=-1)
    
    rewards_batch = 0.5 * local_rewards + 0.5 * (-global_enstrophy[:, None])
    rewards_batch = rewards_batch[..., None] 

    # --- GPU Memory Protection Casts ---
    safe_w_hat = safe_w_hat.astype(jnp.complex128) 
    next_obs_batch_no_pe = next_obs_batch_no_pe.astype(jnp.float32)
    rewards_batch = rewards_batch.astype(jnp.float32)
    dones_batch = dones_batch.astype(jnp.float32)
    
    return safe_w_hat, next_obs_batch_no_pe, rewards_batch, dones_batch

# --- 4. FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, max_steps):
    def step_fn(w_hat, _):
        obs_no_pe = build_marl_obs_batch(w_hat[None, ...]) 
        act = get_batch_actions(actor_params, obs_no_pe, None, add_noise=False)
        act_flat = act.squeeze() 
        
        w_phys_traj, _ = dynamics.unroll_controlled(
            w_hat_init=w_hat, xi_fixed=xi_fixed, params=act_flat, 
            t_steps=1, N_grid=N_GRID, L=L_DOMAIN, dt=DT, 
            substeps=SUBSTEPS, viscosity=VISCOSITY, actuator_grid_shape=(8, 8)
        )
        
        next_w_phys = w_phys_traj[-1]
        next_w_hat = jnp.fft.fft2(next_w_phys)
        
        enstrophy = jnp.mean(next_w_phys**2)
        crashed = jnp.isnan(next_w_phys).any() | jnp.isinf(next_w_phys).any() | (jnp.max(jnp.abs(next_w_phys)) > 100.0)
        
        return next_w_hat, (enstrophy, crashed)

    _, (enstrophies, crashes) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    return jnp.mean(enstrophies), jnp.any(crashes)

# --- Vectorized Training Loop ---
print("Loading 2D Turbulence Initial Conditions...")
data_dir = Path('../../data')
file_path = data_dir / 'turbulence_chaotic_ics_64_more.pkl'

if file_path.exists():
    with open(file_path, 'rb') as f:
        state_bank = jnp.array(pickle.load(f))
    print(f"Loaded {len(state_bank)} ICs from {file_path}")
else:
    print("Generating ICs...")
    state_bank = get_batch_initial_conditions(key, 500, N_GRID, L_DOMAIN)
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(state_bank), f)

# PRE-COMPILATION 
print("Pre-compiling Neural Network graphs for GPU...")
dummy_obs_cat = jnp.zeros((NUM_PARALLEL_ENVS, N_AGENTS, stored_obs_dim))
_ = get_batch_actions(actor_params, dummy_obs_cat, key, add_noise=True)

key, subkey = jax.random.split(key)
w_hat_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))
obs_batch = build_marl_obs_batch(w_hat_batch)
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

python_buffer_size = 0

print(f"Starting Massively Parallel MARL Training (2D Turbulence, N_Envs={NUM_PARALLEL_ENVS})...")
start_time = time.time()

for update_step in range(TOTAL_UPDATES):
    
    if update_step % EVAL_INT == 0:
        eval_u = state_bank[0] 
        eval_e, crashed = fast_eval_episode(actor_params, eval_u, MAX_ENV_STEPS)
        
        actor_norm = jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(actor_params)))
        
        if crashed:
            print(f"Update {update_step:06d} | Enstrophy: [CRASHED] | Actor Norm: {actor_norm:.2f} | Time: {time.time()-start_time:.1f}s")
        else:
            print(f"Update {update_step:06d} | Enstrophy: {eval_e:.6f} | Actor Norm: {actor_norm:.2f} | Time: {time.time()-start_time:.1f}s")

    key, act_key, physics_key, reset_key, sample_key = jax.random.split(key, 5)
    
    if update_step < WARMUP_UPDATES:
        actions = jax.random.uniform(act_key, (NUM_PARALLEL_ENVS, N_AGENTS, 1), minval=-U_MAX, maxval=U_MAX)
    else:
        actions = get_batch_actions(actor_params, obs_batch, act_key, add_noise=True)
        
    next_w_hat_batch, next_obs_batch, rewards_batch, dones_batch = parallel_marl_physics_step(w_hat_batch, actions)
    
    env_step_counts += 1
    truncations_batch = env_step_counts >= MAX_ENV_STEPS
    
    safe_rewards = jnp.where(dones_batch[:, None, None], -100.0, rewards_batch)
    dones_expanded = jnp.tile(dones_batch[:, None, None], (1, N_AGENTS, 1))

    buffer = add_batch_to_buffer(buffer, obs_batch, actions, safe_rewards, next_obs_batch, dones_expanded)
    
    needs_reset = jnp.logical_or(dones_batch.flatten(), truncations_batch)
    if jnp.any(needs_reset):
        fresh_states = jax.random.choice(reset_key, state_bank, shape=(NUM_PARALLEL_ENVS,))
        w_hat_batch = jnp.where(needs_reset[:, None, None], fresh_states, next_w_hat_batch)
        obs_batch = build_marl_obs_batch(w_hat_batch)
        env_step_counts = jnp.where(needs_reset, 0, env_step_counts)
    else:
        w_hat_batch, obs_batch = next_w_hat_batch, next_obs_batch
        
    python_buffer_size = min(python_buffer_size + NUM_PARALLEL_ENVS, 10_000)
    
    if python_buffer_size > ENV_BATCH_SIZE:
        bx, bu, br, bnx, bd = sample_buffer(buffer, ENV_BATCH_SIZE, sample_key) 
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

with open('models/marl_turbulence_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")