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
from models_rl import CentralizedActor, CentralizedCritic
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions

# --- Import solver directly instead of the dual dynamics wrapper ---
import tesseracts.turbulence2d.solver as solver

# --- Configurations ---
N_AGENTS = 64          # 8x8 grid
L_DOMAIN = 1.0         # Domain size
N_GRID = 64
U_MAX = 75.0           # Action scaling for Turbulence

ENV_BATCH_SIZE = 128 
EVAL_INT = 10
POLICY_DELAY = 2 

# Turbulence Specific Control Timing & Physics
MAX_ENV_STEPS = 150    # Control steps (T_steps)
SUBSTEPS = 5           # Physics steps per control step
DT = 0.01              # Physics dt
VISCOSITY = 5e-4       # Fluid viscosity

# Vectorization Configs
NUM_PARALLEL_ENVS = 64
TOTAL_UPDATES = 150000 # 1000 episodes * 150 steps/episode
WARMUP_UPDATES = 500

# --- Initialization ---
key = jax.random.PRNGKey(42)

# --- Precompute Spectral Grid & Forcing Profiles Globally ---
kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_GRID, L_DOMAIN)
dt_phys = DT / SUBSTEPS

grid_dim = int(np.sqrt(N_AGENTS))
x_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
y_c = jnp.linspace(0, L_DOMAIN, grid_dim, endpoint=False) + L_DOMAIN/(2*grid_dim)
xv, yv = jnp.meshgrid(x_c, y_c)
centers_flat = jnp.stack([xv.flatten(), yv.flatten()], axis=1)

# Matches sigma=0.05 from your solver configuration
forcing_hat = solver.compute_forcing_profile(
    centers_flat[:, 0], centers_flat[:, 1], N_GRID, L_DOMAIN, 0.05
)

# Models
actor = CentralizedActor(n_agents=N_AGENTS)
critic = CentralizedCritic()

key, *subkeys = jax.random.split(key, 4)

# Dummy inputs for initialization
dummy_z = jnp.zeros((ENV_BATCH_SIZE, N_GRID, N_GRID))
dummy_action = jnp.zeros((ENV_BATCH_SIZE, N_AGENTS, 1))

actor_params = actor.init(subkeys[0], dummy_z)
critic_params = critic.init(subkeys[1], dummy_z, dummy_action)

target_actor_params = actor_params
target_critic_params = critic_params

tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# --- 1. ON-DEVICE REPLAY BUFFER ---
@struct.dataclass
class DeviceReplayBuffer:
    z: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    nz: jnp.ndarray
    d: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, n_grid, n_agents, a_dim):
        return cls(
            z=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float64),
            a=jnp.zeros((max_size, n_agents, a_dim), dtype=jnp.float64),
            r=jnp.zeros((max_size, 1), dtype=jnp.float64),
            nz=jnp.zeros((max_size, n_grid, n_grid), dtype=jnp.float64),
            d=jnp.zeros((max_size, 1), dtype=jnp.float64),
            ptr=jnp.int32(0),
            size=jnp.int32(0),
            max_size=max_size
        )

@jax.jit
def add_batch_to_buffer(buffer, z_b, a_b, r_b, nz_b, d_b):
    batch_size = z_b.shape[0]
    indices = (buffer.ptr + jnp.arange(batch_size)) % buffer.max_size
    
    new_z = buffer.z.at[indices].set(z_b)
    new_a = buffer.a.at[indices].set(a_b)
    new_r = buffer.r.at[indices].set(r_b)
    new_nz = buffer.nz.at[indices].set(nz_b)
    new_d = buffer.d.at[indices].set(d_b)
    
    new_ptr = (buffer.ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)
    
    return buffer.replace(z=new_z, a=new_a, r=new_r, nz=new_nz, d=new_d, ptr=new_ptr, size=new_size)

@partial(jax.jit, static_argnames=['batch_size'])
def sample_buffer(buffer, batch_size, key):
    valid_range = jnp.minimum(buffer.size, buffer.max_size)
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=valid_range)
    return buffer.z[indices], buffer.a[indices], buffer.r[indices], buffer.nz[indices], buffer.d[indices]

buffer = DeviceReplayBuffer.create(12_500, N_GRID, N_AGENTS, 1)


# --- 3. JIT TRAINING & ROLLOUT FUNCTIONS ---
@jax.jit
def update_critic(c_p, ta_p, tc_p, opt_c, z, a, r, nz, d, key): 
    key, noise_key = jax.random.split(key)
    
    noise = jnp.clip(jax.random.normal(noise_key, a.shape) * (U_MAX * 0.1), -U_MAX * 0.5, U_MAX * 0.5)
    next_a = jnp.clip(actor.apply(ta_p, nz) + noise, -U_MAX, U_MAX)
    
    t_q1, t_q2 = critic.apply(tc_p, nz, next_a)
    target_q = r + 0.99 * (1.0 - d) * jnp.minimum(t_q1, t_q2)
    
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, z, a)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    return optax.apply_updates(c_p, up_c), opt_c, l_c # ---> RETURN l_c

@jax.jit
def update_actor_and_targets(a_p, c_p, ta_p, tc_p, opt_a, z):
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, z, actor.apply(p, z))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    a_p = optax.apply_updates(a_p, up_a)
    
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda new, old: tau*new + (1-tau)*old, c_p, tc_p)
    return a_p, new_ta, new_tc, opt_a, l_a # ---> RETURN l_a

@partial(jax.jit, static_argnames=['add_noise'])
def get_batch_actions(a_p, z_batch, key, add_noise=True):
    actions = jax.vmap(actor.apply, in_axes=(None, 0))(a_p, z_batch)
    
    if add_noise:
        noise = jax.random.normal(key, actions.shape) * (U_MAX * 0.1)
        actions = jnp.clip(actions + noise, -U_MAX, U_MAX)
    return actions

@jax.jit
def parallel_physics_step(w_init_batch, actions):
    acts_flat = actions.squeeze(-1) 
    
    def single_physics_step(w_single, act_single):
        # 1. Convert physical to spectral space
        w_hat = jnp.fft.fft2(w_single)
        
        # 2. Run RK4 directly using precomputed constants
        def rk4_loop(i, w):
            return solver.rk4_step(
                w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_single
            )
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        
        # 3. Convert back to physical space and return
        return jnp.fft.ifft2(w_hat_next).real
    
    next_w_batch = jax.vmap(single_physics_step)(w_init_batch, acts_flat)
    
    is_invalid = jnp.logical_not(jnp.isfinite(next_w_batch).all(axis=(1, 2)))
    is_exploding = jnp.max(jnp.abs(next_w_batch), axis=(1, 2)) > 1000.0 
    dones_batch = jnp.logical_or(is_invalid, is_exploding)[:, None]
    
    safe_w = jnp.where(dones_batch[:, :, None], jnp.zeros_like(next_w_batch), next_w_batch)
    
    global_enstrophy = jnp.mean(jnp.square(safe_w), axis=(1, 2))[:, None]
    rewards_batch = -global_enstrophy
    
    return safe_w, rewards_batch, dones_batch

# --- 4. FAST JIT-COMPILED EVALUATION ---
@partial(jax.jit, static_argnames=['max_steps'])
def fast_eval_episode(actor_params, init_state, max_steps):
    def step_fn(state, _):
        act = actor.apply(actor_params, state[None, ...])
        act_flat = act.squeeze() 
        
        # 1. Convert to spectral space
        w_hat = jnp.fft.fft2(state)
        
        # 2. Run RK4 directly
        def rk4_loop(i, w):
            return solver.rk4_step(
                w, dt_phys, kx, ky, k_sq, k_inv, VISCOSITY, forcing_hat, act_flat
            )
        w_hat_next = jax.lax.fori_loop(0, SUBSTEPS, rk4_loop, w_hat)
        
        # 3. Convert back to physical space
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

if jnp.iscomplexobj(state_bank):
    print("Converting spectral initial conditions to physical space...")
    state_bank = jnp.fft.ifft2(state_bank).real

key, subkey = jax.random.split(key)
w_batch = jax.random.choice(subkey, state_bank, shape=(NUM_PARALLEL_ENVS,))
env_step_counts = jnp.zeros(NUM_PARALLEL_ENVS)

python_buffer_size = 0

print("Starting Massively Parallel Centralized RL Training (2D Turbulence)...")
start_time = time.time()

# --- Initialize loss tracking variables ---
actor_loss_val = 0.0
critic_loss_val = 0.0

for update_step in trange(TOTAL_UPDATES):
    
    if update_step % EVAL_INT == 0:
        eval_w = state_bank[0] 
        eval_enstrophy, crashed = fast_eval_episode(actor_params, eval_w, MAX_ENV_STEPS)
        episode_num = update_step // MAX_ENV_STEPS
        
        # --- Telemetry Calculation ---
        # Calculate mean absolute action to ensure it's doing something
        # (Define actions securely for the first loop pass)
        mean_act = jnp.mean(jnp.abs(actions)) if update_step > 0 else 0.0
        
        if crashed:
            print(f"Upd {update_step:05d} | Ep {episode_num} | Eval: [CRASHED] | Act: {mean_act:.2f} | a_loss: {actor_loss_val:.2f} | c_loss: {critic_loss_val:.2f}")
        else:
            print(f"Upd {update_step:05d} | Ep {episode_num} | Eval: {eval_enstrophy:.4f} | Act: {mean_act:.2f} | a_loss: {actor_loss_val:.2f} | c_loss: {critic_loss_val:.2f}")

    # 2. Parallel Data Collection 
    key, act_key, reset_key = jax.random.split(key, 3)
    
    if update_step < WARMUP_UPDATES:
        actions = jax.random.uniform(act_key, (NUM_PARALLEL_ENVS, N_AGENTS, 1), minval=-U_MAX*0.01, maxval=U_MAX*0.01)
    else:
        actions = get_batch_actions(actor_params, w_batch, act_key, add_noise=True)
        
    next_w_batch, rewards_batch, dones_batch = parallel_physics_step(w_batch, actions)
    
    env_step_counts += 1
    truncations_batch = env_step_counts >= MAX_ENV_STEPS
    
    safe_rewards = jnp.where(dones_batch, -500.0, rewards_batch)

    buffer = add_batch_to_buffer(buffer, w_batch, actions, safe_rewards, next_w_batch, dones_batch)
    
    # 4. Handle Resets 
    needs_reset = jnp.logical_or(dones_batch.flatten(), truncations_batch)
    fresh_states = jax.random.choice(reset_key, state_bank, shape=(NUM_PARALLEL_ENVS,))
    
    w_batch = jnp.where(needs_reset[:, None, None], fresh_states, next_w_batch)
    env_step_counts = jnp.where(needs_reset, 0, env_step_counts)
        
    # 5. TD3 Updates
    python_buffer_size = min(python_buffer_size + NUM_PARALLEL_ENVS, 12_500)
    
    if python_buffer_size > ENV_BATCH_SIZE:
        bz, ba, br, bnz, bd = sample_buffer(buffer, ENV_BATCH_SIZE, subkey) 
        key, subkey = jax.random.split(key)
        
        # --- Catch Critic Loss ---
        critic_params, opt_critic, critic_loss_val = update_critic(
            critic_params, target_actor_params, target_critic_params, opt_critic, bz, ba, br, bnz, bd, subkey
        )
        
        if update_step % POLICY_DELAY == 0:
            # ---> Catch Actor Loss ---
            actor_params, target_actor_params, target_critic_params, opt_actor, actor_loss_val = update_actor_and_targets(
                actor_params, critic_params, target_actor_params, target_critic_params, opt_actor, bz
            )

# Save
with open('models/rl_turb_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params}))
print(f"Training finished in {time.time()-start_time:.1f}s. Weights saved.")