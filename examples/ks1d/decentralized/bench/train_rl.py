import jax
import jax.numpy as jnp
import optax
import flax.serialization
import numpy as np
import time
from pathlib import Path
import jax.tree_util
import sys

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

# Project imports
from models_rl import CentralizedActor, CentralizedCritic
from examples.ks1d.decentralized.data_utils import evolve_to_attractor
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 

# --- Configurations ---
N_AGENTS = 8
L_DOMAIN = 22.0
N_GRID = 128
BATCH_SIZE = 32
MAX_EPISODES = 500
WARMUP_EPISODES = 25
MAX_STEPS = 200
EVAL_INT = 10

# --- Initialization ---
key = jax.random.PRNGKey(42)

def direct_control_policy(action_params, u_obs, u_target, xi_fixed):
    return action_params

dynamics = PDEDynamics(policy_apply_fn=direct_control_policy)
xi_fixed = jnp.linspace(0.0, L_DOMAIN, N_AGENTS, endpoint=False) + (L_DOMAIN/N_AGENTS)/2

# Models
actor = CentralizedActor(n_agents=N_AGENTS)
critic = CentralizedCritic()

key, *subkeys = jax.random.split(key, 4)
dummy_state = jnp.zeros((BATCH_SIZE, N_GRID))
dummy_action = jnp.zeros((BATCH_SIZE, N_AGENTS))

actor_params = actor.init(subkeys[0], dummy_state)
critic_params = critic.init(subkeys[1], dummy_state, dummy_action)
target_actor_params, target_critic_params = actor_params, critic_params

# Optimizers
tx_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-5))
opt_actor = tx_actor.init(actor_params)
opt_critic = tx_critic.init(critic_params)

# Standard Replay Buffer (Stores global state vectors)
class GlobalReplayBuffer:
    def __init__(self, max_size, s_dim, a_dim):
        self.s = np.zeros((max_size, s_dim))
        self.a = np.zeros((max_size, a_dim))
        self.r = np.zeros((max_size, 1))
        self.ns = np.zeros((max_size, s_dim))
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def add(self, s, a, r, ns):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return jnp.array(self.s[ind]), jnp.array(self.a[ind]), \
               jnp.array(self.r[ind]), jnp.array(self.ns[ind])

buffer = GlobalReplayBuffer(100000, N_GRID, N_AGENTS)

# --- JIT Training Functions ---

@jax.jit
def train_step(a_p, c_p, ta_p, tc_p, opt_a, opt_c, s, a, r, ns, key):
    # TD3 Target logic
    key, noise_key = jax.random.split(key)
    noise = jnp.clip(jax.random.normal(noise_key, a.shape) * 0.2, -0.5, 0.5)
    next_a = jnp.clip(actor.apply(ta_p, ns) + noise, -1.0, 1.0)
    
    t_q1, t_q2 = critic.apply(tc_p, ns, next_a)
    target_q = r + 0.99 * jnp.minimum(t_q1, t_q2)
    
    # Critic Update
    def c_loss_fn(p):
        q1, q2 = critic.apply(p, s, a)
        return jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
    
    l_c, grads_c = jax.value_and_grad(c_loss_fn)(c_p)
    up_c, opt_c = tx_critic.update(grads_c, opt_c)
    
    # Actor Update
    def a_loss_fn(p):
        return -jnp.mean(critic.apply(c_p, s, actor.apply(p, s))[0])
    
    l_a, grads_a = jax.value_and_grad(a_loss_fn)(a_p)
    up_a, opt_a = tx_actor.update(grads_a, opt_a)
    
    # Soft Updates
    tau = 0.005
    new_ta = jax.tree_util.tree_map(lambda x, y: tau*x + (1-tau)*y, a_p, ta_p)
    new_tc = jax.tree_util.tree_map(lambda x, y: tau*x + (1-tau)*y, c_p, tc_p)
    
    return optax.apply_updates(a_p, up_a), optax.apply_updates(c_p, up_c), \
           new_ta, new_tc, opt_a, opt_c

# --- Training Loop ---
for ep in range(MAX_EPISODES):
    # Eval
    if ep % EVAL_INT == 0:
        u = evolve_to_attractor(jax.random.PRNGKey(ep), N_GRID, L_DOMAIN)
        eval_e = 0.0
        for _ in range(MAX_STEPS):
            action = actor.apply(actor_params, u)
            traj = dynamics.unroll_controlled(u, xi_fixed, jnp.zeros(N_GRID), action, 1, N_grid=N_GRID, L=L_DOMAIN)
            u = traj[0][-1]
            eval_e += jnp.mean(u**2)
        print(f"Eval Ep {ep:03d} | Global Energy: {eval_e/MAX_STEPS:.6f}")

    # Real Train
    key, subkey = jax.random.split(key)
    u = evolve_to_attractor(subkey, N_GRID, L_DOMAIN)
    for t in range(MAX_STEPS):
        if ep < WARMUP_EPISODES:
            action = np.random.uniform(-1.0, 1.0, (N_AGENTS,))
        else:
            action = np.array(actor.apply(actor_params, u))
        
        traj = dynamics.unroll_controlled(u, xi_fixed, jnp.zeros(N_GRID), action, 1, N_grid=N_GRID, L=L_DOMAIN)
        next_u = traj[0][-1]
        
        # Kill switch
        if jnp.isnan(next_u).any(): break
        
        reward = -jnp.mean(next_u**2)
        buffer.add(u, action, reward, next_u)
        u = next_u
        
        if buffer.size > BATCH_SIZE:
            bs, ba, br, bns = buffer.sample(BATCH_SIZE)
            key, subkey = jax.random.split(key)
            actor_params, critic_params, target_actor_params, target_critic_params, opt_actor, opt_critic = \
                train_step(actor_params, critic_params, target_actor_params, target_critic_params, 
                           opt_actor, opt_critic, bs, ba, br, bns, subkey)

# Save
with open('rl_centralized_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'actor': actor_params}))
print("RL training finished and weights saved to rl_centralized_params.msgpack.")
