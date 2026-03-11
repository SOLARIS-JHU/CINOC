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
from tqdm import trange

# Enable x64 for Spectral Stability (Crucial for Turbulence)
jax.config.update("jax_enable_x64", True)

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from env_turb2d import Turbulence2DMARLEnv, extract_patches_2d_jit
from models_marl import MARLActor2D, MARLCritic2D
from utils_hypemarl import get_sinusoidal_encoding
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions

# Import solver directly
import tesseracts.turbulence2d.solver as solver

# --- Configurations ---
N_AGENTS = 64          # 8x8 grid
L_DOMAIN = 1.0         # Domain size
N_GRID = 64
U_MAX = 75.0           # Action scaling for Turbulence

ENV_BATCH_SIZE = 128 
NN_BATCH_SIZE = 512    # Subsampled Neural Network Batch Size
EVAL_INT = 10
POLICY_DELAY = 2 

# Turbulence Specific Control Timing
MAX_ENV_STEPS = 150    # Control steps (T_steps)
SUBSTEPS = 5           # Physics steps per control step
DT = 0.01              # Physics dt
VISCOSITY = 5e-4       # Fluid viscosity
SIGMA = 0.05           # Actuator Gaussian spread

# Vectorization Configs
NUM_PARALLEL_ENVS = 64
TOTAL_UPDATES = 100000 
WARMUP_UPDATES = 500

def get_2d_sinusoidal_encoding(p_2d, d=1024, n=1000.0):
    """Combines two 1D positional encodings for 2D coordinates."""
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

# --- Initialization ---
key = jax.random.PRNGKey(42)

# Precompute Spectral Grid & Forcing Profiles Globally
kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_GRID, L_DOMAIN)
dt_phys = DT / SUBSTEPS

grid_dim = int(np.sqrt(N_AGENTS))
x_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
y_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
xv, yv = jnp.meshgrid(x_c, y_c)
centers_flat = jnp.stack([xv.flatten(), yv.flatten()], axis=1)

forcing_hat = solver.compute_forcing_profile(
    centers_flat[:, 0], centers_flat[:, 1], N_GRID, L_DOMAIN, SIGMA
)

# Initialize dummy env to extract static parameters
dummy_pool = jnp.zeros((1, N_GRID, N_GRID))
env = Turbulence2DMARLEnv(
    initial_conditions=dummy_pool, n_agents=N_AGENTS, 
    N_grid=N_GRID, L=L_DOMAIN, dt=DT, viscosity=VISCOSITY, 
    substeps=SUBSTEPS, max_steps=MAX_ENV_STEPS, sigma=SIGMA
)

patch_size = env.patch_size 
local_y_dim = env.local_y_dim 
n_mu = env.n_mu 
pe_dim = 2048 # 1024 * 2 (X and Y)

stored_obs_dim = local_y_dim + n_mu 
total_input_dim = stored_obs_dim + pe_dim

# Extract static variables for JAX
xi_fixed = jnp.array(env.agent_positions)
xi_norm = jnp.array(env.xi_norm)

# Explicitly cast static arrays to float32 to prevent 64-bit upcasting in NN
mu_jax = jnp.array(env.mu, dtype=jnp.float32)
pe_jax = jnp.array(get_2d_sinusoidal_encoding(xi_norm, d=1024), dtype=jnp.float32)
target_state = jnp.zeros((N_GRID, N_GRID), dtype=jnp.float32)

actor = MARLActor2D()
critic = MARLCritic2D()

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

@partial(jax.jit, donate_argnums=(0,))
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
    
    noise = jnp.clip(jax.random.normal(noise_key, u.shape) * (U_MAX * 0.1), -U_MAX * 0.5, U_MAX * 0.5)
    next_u = jnp.clip(actor.apply(ta_p, nx) + noise, -U_MAX, U_MAX)
    
    t_q1, t_q2 = critic.apply(tc_p, nx, next_u)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, x, u)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c, l_c

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
    return a_p, new_ta, new_tc, opt_a, l_a

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
def parallel_marl_physics_step(w_init_batch, actions):
    acts_flat = actions.squeeze(-1) 
    
    def single_physics_step(w_single, act_single):
        w_hat = jnp.fft.fft2(w_single)
        def rk4_loop(i, w):
            return solver.rk4_step(
                w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_single
            )
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        return jnp.fft.ifft2(w_hat_next).real
    
    next_w_batch = jax.vmap(single_physics_step)(w_init_batch, acts_flat)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_w_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_w_batch), axis=(1, 2)) > 1000.0
    dones_batch = jnp.logical_or(is_invalid, is_exploding)
    
    safe_w = jnp.where(dones_batch[:, None, None], jnp.zeros_like(next_w_batch), next_w_batch)
    next_obs_batch_no_pe = build_marl_obs_batch(safe_w)
    
    # Enstrophy Reward Logic
    global_enstrophy = jnp.mean(jnp.square(safe_w), axis=(1, 2))
    
    # Local Stability
    y_local_err = next_obs_batch_no_pe[..., :patch_size**2] 
    local_rewards = -jnp.mean(jnp.square(y_local_err), axis=-1)
    action_penalty = -1e-3 * jnp.mean(jnp.square(actions), axis=-1)

    rewards_batch = 0.5 * local_rewards + 0.5 * (-global_enstrophy[:, None]) + action_penalty
    rewards_batch = rewards_batch[..., None] 
    
    return safe_w, next_obs_batch_no_pe, rewards_batch, dones_batch

# --- 4. FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, max_steps):
    def step_fn(state, _):
        obs_no_pe = build_marl_obs_batch(state[None, ...]) 
        act = get_batch_actions(actor_params, obs_no_pe, None, add_noise=False)
        act_flat = act.squeeze() 
        
        w_hat = jnp.fft.fft2(state)
        def rk4_loop(i, w):
            return solver.rk4_step(
                w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_flat
            )
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        next_state = jnp.fft.ifft2(w_hat_next).real
        
        enstrophy = jnp.mean(next_state**2)
        crashed = jnp.isnan(next_state).any() | jnp.isinf(next_state).any() | (jnp.max(jnp.abs(next_state)) > 1000.0)
        
        return next_state, (enstrophy, crashed)

    _, (enstrophies, crashes) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    return jnp.mean(enstrophies), jnp.any(crashes)

# --- Vectorized Training Loop ---
print("Loading 2D Turbulence Initial Conditions...")
data_dir = Path('../../data')
data_dir.mkdir(parents=True, exist_ok=True)
file_path = data_dir / 'turbulence_chaotic_ics_64_more.pkl'

if file_path.exists():
    with open(file_path, 'rb') as f:
        state_bank = jnp.array(pickle.load(f))
    print(f"Loaded {len(state_bank)} ICs from {file_path}")
else:
    print("Generating ICs (this may take a few minutes)...")
    state_bank = get_batch_initial_conditions(key, 500, N_GRID, L_DOMAIN)
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(state_bank), f)

# Convert from spectral if needed
if jnp.iscomplexobj(state_bank):
    print("Converting spectral initial conditions to physical space...")
    state_bank = jnp.fft.ifft2(state_bank).real

key, subkey = jax.random.split(key)
w_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))
obs_batch = build_marl_obs_batch(w_batch)
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

python_buffer_size = 0

print("Starting Massively Parallel MARL Training (2D Turbulence)...")
start_time = time.time()

actor_loss_val = 0.0
critic_loss_val = 0.0

for update_step in trange(TOTAL_UPDATES):
    
    if update_step % EVAL_INT == 0:
        eval_w = state_bank[0] 
        eval_e, crashed = fast_eval_episode(actor_params, eval_w, MAX_ENV_STEPS)
        episode_num = update_step // MAX_ENV_STEPS
        
        mean_act = jnp.mean(jnp.abs(actions)) if update_step > 0 else 0.0
        
        if crashed:
            print(f"Upd {update_step:05d} | Ep {episode_num} | Eval Enstrophy: [CRASHED] | Act: {mean_act:.2f} | a_loss: {actor_loss_val:.2f} | c_loss: {critic_loss_val:.2f}")
        else:
            print(f"Upd {update_step:05d} | Ep {episode_num} | Eval Enstrophy: {eval_e:.4f} | Act: {mean_act:.2f} | a_loss: {actor_loss_val:.2f} | c_loss: {critic_loss_val:.2f}")

    # 2. Parallel Data Collection 
    key, act_key, reset_key = jax.random.split(key, 3)
    
    if update_step < WARMUP_UPDATES:
        actions = jax.random.uniform(act_key, (NUM_PARALLEL_ENVS, N_AGENTS, 1), minval=-U_MAX*0.01, maxval=U_MAX*0.01)
    else:
        actions = get_batch_actions(actor_params, obs_batch, act_key, add_noise=True)
        
    next_w_batch, next_obs_batch, rewards_batch, dones_batch = parallel_marl_physics_step(w_batch, actions)
    
    env_step_counts += 1
    truncations_batch = env_step_counts >= MAX_ENV_STEPS
    
    safe_rewards = jnp.where(dones_batch[:, None, None], -500.0, rewards_batch)
    dones_expanded = jnp.tile(dones_batch[:, None, None], (1, N_AGENTS, 1))

    buffer = add_batch_to_buffer(buffer, obs_batch, actions, safe_rewards, next_obs_batch, dones_expanded)
    
    # 4. Handle Resets 
    needs_reset = jnp.logical_or(dones_batch.flatten(), truncations_batch)
    fresh_states = jax.random.choice(reset_key, state_bank, shape=(NUM_PARALLEL_ENVS,))
    
    w_batch = jnp.where(needs_reset[:, None, None], fresh_states, next_w_batch)
    
    obs_batch = build_marl_obs_batch(w_batch)
    env_step_counts = jnp.where(needs_reset, 0, env_step_counts)
        
    # 5. TD3 Updates
    python_buffer_size = min(python_buffer_size + NUM_PARALLEL_ENVS, 12_500)
    
    if python_buffer_size > ENV_BATCH_SIZE:
        bx, bu, br, bnx, bd = sample_buffer(buffer, ENV_BATCH_SIZE, subkey) 
        key, subkey = jax.random.split(key)
        
        # Flatten the 128 x 64 arrays
        bx_flat = bx.reshape(-1, stored_obs_dim)
        bu_flat = bu.reshape(-1, 1)
        br_flat = br.reshape(-1, 1)
        bnx_flat = bnx.reshape(-1, stored_obs_dim)
        bd_flat = bd.reshape(-1, 1)
        
        # Agent Subsampling 
        idx = jax.random.randint(subkey, shape=(NN_BATCH_SIZE,), minval=0, maxval=bx_flat.shape[0])
        
        # Get the specific agent index (0-63) for each chosen sample to fetch the right PE
        agent_indices = idx % N_AGENTS
        pe_sub = pe_jax[agent_indices] 
        
        # Build the final small batch of 512 samples
        bx_full = jnp.concatenate([bx_flat[idx], pe_sub], axis=-1)
        bnx_full = jnp.concatenate([bnx_flat[idx], pe_sub], axis=-1)
        bu_sub = bu_flat[idx]
        br_sub = br_flat[idx]
        bd_sub = bd_flat[idx]
        
        critic_params, opt_critic, critic_loss_val = update_critic(
            critic_params, target_actor_params, target_critic_params, opt_critic, 
            bx_full, bu_sub, br_sub, bnx_full, bd_sub, subkey
        )
        
        if update_step % POLICY_DELAY == 0:
            actor_params, target_actor_params, target_critic_params, opt_actor, actor_loss_val = update_actor_and_targets(
                actor_params, critic_params, target_actor_params, target_critic_params, opt_actor, bx_full
            )

# Save
with open('models/marl_turb_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")