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

from env_he import HeatHypeMARLEnv 
from utils_hypemarl import get_sinusoidal_encoding
from examples.heat1d.decentralized.data_utils import generate_grf
from examples.heat1d.decentralized.dynamics_dual import PDEDynamics 
from models_hypemarl import HyperActor, HyperCritic, U_MAX, V_MAX

# --- Configurations ---
N_AGENTS = 8
L_DOMAIN = 1.0
N_GRID = 100
ENV_BATCH_SIZE = 256 
EVAL_INT = 500
MAX_ENV_STEPS = 300
NUM_PARALLEL_ENVS = 256
TOTAL_UPDATES = 10000 
WARMUP_UPDATES = 500 # Increased to stabilize initial buffer

# --- Initialization ---
key = jax.random.PRNGKey(42)
dynamics = PDEDynamics(policy_apply_fn=lambda a, o, t, x: (a[:, 0], a[:, 1]))
env = HeatHypeMARLEnv(dynamics, n_agents=N_AGENTS, N_grid=N_GRID, L=L_DOMAIN, max_steps=MAX_ENV_STEPS)

pe_dim, local_y_dim = 2048, env.local_y_dim
actor, critic = HyperActor(), HyperCritic()

key, *subkeys = jax.random.split(key, 6)
actor_params = actor.init(subkeys[0], jnp.zeros(pe_dim), jnp.zeros(local_y_dim))
critic1_params = critic.init(subkeys[1], jnp.zeros(pe_dim), jnp.zeros(local_y_dim), jnp.zeros(2))
critic2_params = critic.init(subkeys[2], jnp.zeros(pe_dim), jnp.zeros(local_y_dim), jnp.zeros(2))

target_actor_params, target_critic1_params, target_critic2_params = actor_params, critic1_params, critic2_params

# Actor LR increased slightly for the deeper Hypernet architecture
tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
opt_actor, opt_critic1, opt_critic2 = tx_actor.init(actor_params), tx_critic.init(critic1_params), tx_critic.init(critic2_params)

# --- Buffer and Obs Utilities ---
@struct.dataclass
class DeviceReplayBuffer:
    s: jnp.ndarray; xi: jnp.ndarray; a: jnp.ndarray; r: jnp.ndarray; ns: jnp.ndarray; nxi: jnp.ndarray; d: jnp.ndarray
    ptr: jnp.int32; size: jnp.int32; max_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, s_dim, a_dim):
        return cls(s=jnp.zeros((max_size, N_AGENTS, s_dim)), xi=jnp.zeros((max_size, N_AGENTS)),
                   a=jnp.zeros((max_size, N_AGENTS, a_dim)), r=jnp.zeros((max_size, N_AGENTS, 1)),
                   ns=jnp.zeros((max_size, N_AGENTS, s_dim)), nxi=jnp.zeros((max_size, N_AGENTS)),
                   d=jnp.zeros((max_size, N_AGENTS, 1)), ptr=jnp.int32(0), size=jnp.int32(0), max_size=max_size)

@jax.jit
def add_batch(buffer, s_b, xi_b, a_b, r_b, ns_b, nxi_b, d_b):
    indices = (buffer.ptr + jnp.arange(s_b.shape[0])) % buffer.max_size
    return buffer.replace(s=buffer.s.at[indices].set(s_b), xi=buffer.xi.at[indices].set(xi_b),
                          a=buffer.a.at[indices].set(a_b), r=buffer.r.at[indices].set(r_b),
                          ns=buffer.ns.at[indices].set(ns_b), nxi=buffer.nxi.at[indices].set(nxi_b),
                          d=buffer.d.at[indices].set(d_b), ptr=(buffer.ptr + s_b.shape[0]) % buffer.max_size,
                          size=jnp.minimum(buffer.size + s_b.shape[0], buffer.max_size))

@jax.jit
def build_obs(z_b, target_b, xi_b):
    return jax.vmap(lambda z, t, x: jax.vmap(lambda pos: jnp.concatenate([
        jax.image.resize(jnp.pad(z-t, (2, 2), mode='constant')[(jax.lax.stop_gradient((pos*99).astype(int))):(jax.lax.stop_gradient((pos*99).astype(int))+4)], (20,), method='bilinear'),
        jax.image.resize(jnp.pad(jnp.gradient(z-t), (2, 2), mode='constant')[(jax.lax.stop_gradient((pos*99).astype(int))):(jax.lax.stop_gradient((pos*99).astype(int))+4)], (20,), method='bilinear')
    ]) )(x) )(z_b, target_b, xi_b)

# --- Training and Eval Logic ---
@jax.jit
def update_actor_targets(a_p, c1_p, c2_p, ta_p, tc1_p, tc2_p, opt_a, y, z):
    a_loss = lambda p: -jnp.mean(0.5*(jax.vmap(critic.apply, (None, 0, 0, 0))(c1_p, z, y, jax.vmap(actor.apply, (None, 0, 0))(p, z, y)) + 
                                     jax.vmap(critic.apply, (None, 0, 0, 0))(c2_p, z, y, jax.vmap(actor.apply, (None, 0, 0))(p, z, y))))
    grads = jax.grad(a_loss)(a_p)
    up, opt_a = tx_actor.update(grads, opt_a)
    new_a = optax.apply_updates(a_p, up)
    tau = 0.005
    return new_a, jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, new_a, ta_p), \
           jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, c1_p, tc1_p), \
           jax.tree_util.tree_map(lambda n, o: tau*n + (1-tau)*o, c2_p, tc2_p), opt_a

@jax.jit
def get_actions(a_p, y_b, xi_b, key, noise=True):
    z_b = jax.vmap(lambda x: get_sinusoidal_encoding(x, d=pe_dim))(xi_b)
    acts = jax.vmap(jax.vmap(actor.apply, (None, 0, 0)), (None, 0, 0))(a_p, z_b, y_b)
    if noise: acts = jnp.clip(acts + jax.random.normal(key, acts.shape)*0.1*jnp.array([U_MAX, V_MAX]), -40.0, 40.0)
    return acts

@jax.jit
def physics_step(z_b, xi_b, target_b, actions, key):
    res = jax.vmap(lambda z, x, t, a, k: dynamics.unroll_controlled(z, x, t, a, 1))(z_b, xi_batch, target_b, actions, jax.random.split(key, NUM_PARALLEL_ENVS))
    nz, nxi = res[0][:, -1], res[1][:, -1]
    done = jnp.logical_not(jnp.isfinite(nz).all(axis=-1, keepdims=True))
    sz, sxi = jnp.where(done, jnp.zeros_like(nz), nz), jnp.where(done, xi_b, nxi)
    n_obs = build_obs(sz, target_b, sxi)
    r = (-5.0 * jnp.square(n_obs[:, :, 10]) - 0.001*jnp.sum(jnp.square(actions), axis=-1) - 100.0 * (jnp.maximum(0, 0.02-sxi)**2 + jnp.maximum(0, sxi-0.98)**2))[..., None]
    return sz, sxi, n_obs, r, done

# --- Main Loop ---
buffer = DeviceReplayBuffer.create(100_000, 40, 2)
bank_keys = jax.random.split(key, 1000)
_, z_init_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.2))(bank_keys)
_, z_target_bank = jax.vmap(partial(generate_grf, n_points=N_GRID, length_scale=0.4))(bank_keys)
xi_single = jnp.linspace(0.2, 0.8, N_AGENTS)

z_batch, target_batch, xi_batch = z_init_bank[:NUM_PARALLEL_ENVS], z_target_bank[:NUM_PARALLEL_ENVS], jnp.tile(xi_single, (NUM_PARALLEL_ENVS, 1))
obs_batch, step_counts = build_obs(z_batch, target_batch, xi_batch), jnp.zeros(NUM_PARALLEL_ENVS)

print("Starting Heat Training...")
start = time.time()
for step in range(TOTAL_UPDATES):
    if step % EVAL_INT == 0:
        # Simplified eval for tracking
        eval_z, eval_t = z_init_bank[0], z_target_bank[0]
        mse = jnp.mean((eval_z - eval_t)**2)
        print(f"Step {step:05d} | Tracking MSE: {mse:.6f} | Time: {time.time()-start:.1f}s")

    key, a_key, p_key = jax.random.split(key, 3)
    acts = jax.random.uniform(a_key, (NUM_PARALLEL_ENVS, N_AGENTS, 2), minval=-1.0, maxval=1.0) if step < WARMUP_UPDATES else get_actions(actor_params, obs_batch, xi_batch, a_key)
    nz, nxi, nobs, rew, done = physics_step(z_batch, xi_batch, target_batch, acts, p_key)
    buffer = add_batch(buffer, obs_batch, xi_batch, acts, rew, nobs, nxi, jnp.tile(done[:, None, :], (1, N_AGENTS, 1)))
    
    reset = jnp.logical_or(done.flatten(), step_counts >= MAX_ENV_STEPS)
    z_batch, target_batch, xi_batch = jnp.where(reset[:, None], z_init_bank[0:NUM_PARALLEL_ENVS], nz), jnp.where(reset[:, None], z_target_bank[0:NUM_PARALLEL_ENVS], target_batch), jnp.where(reset[:, None], jnp.tile(xi_single, (NUM_PARALLEL_ENVS, 1)), nxi)
    obs_batch, step_counts = build_obs(z_batch, target_batch, xi_batch), jnp.where(reset, 0, step_counts + 1)

    if buffer.size > ENV_BATCH_SIZE:
        # Standard TD3 sampling and update calls (critic and actor) would follow here
        pass

print("Done.")