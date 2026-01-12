"""
Centralized Training for Nodal-Lefty Pattern Control

Train a policy network to control pattern morphing in the
Nodal-Lefty reaction-diffusion system.
"""

import sys
from pathlib import Path
import os

# Prevent JAX from preallocating all GPU memory (helps with OOM/fragmentation)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

import jax
import jax.numpy as jnp
import optax
import time
from functools import partial
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import flax.serialization
import numpy as np

from models.policy_schnakenberg import SchnakenbergControlNet
from examples.rdp2d.centralized.dynamics import unroll_with_loss, unroll_jit
from tesseracts.RDP2d.nodal_lefty_solver import (
    NodalLeftyConfig,
    build_imex_operators_neumann,
    imex_step_nodal_lefty,
)


# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# Load Configuration & Data
# =============================================================================

data_dir = Path(__file__).parent.parent / 'data'
if not data_dir.exists():
    print(f"Data not found at {data_dir}"); sys.exit(1)

# Load config
config = np.load(data_dir / 'config.npz')
N_grid = int(config['N'])
L_domain = float(config['L'])
D_n = float(config['D_n'])
D_l = float(config['D_l'])
gamma_n = float(config['gamma_n'])
gamma_l = float(config['gamma_l'])
n_n = float(config['n_n'])
n_l = float(config['n_l'])
k_n = float(config['k_n'])
k_l = float(config['k_l'])
beta_n = float(config['beta_n'])
beta_l = float(config['beta_l'])
dt = float(config['dt'])

print(f"Loaded config: L={L_domain}, N={N_grid}, dt={dt}")

# Alpha values for control formulation (Option A: use INITIAL pattern's alpha)
# The controller must fight against the initial attractor to reach target
alpha_n_initial = 0.8   # From striped pattern (data was generated with this)
alpha_l_initial = 4.0
alpha_n_target = 0.5    # Target spotted pattern
alpha_l_target = 4.0

# Other parameters
n_agents = 25
sigma = 20.0  # Match data config (Gaussian spread in μm)

# Limits
u_max = 10.0
V_max = 10.0

# Training Hyperparameters - MODIFIED for Stability
# 1. Shorter horizon to make learning easier (curriculum could strictly increase this)
T_steps = 200    # Start with 100 hours (easier credit assignment)
# 2. Larger batch size for stable gradients (now possible with checkpointing)
batch_size = 8   
epochs = 1000

# Loss weights (loss is now normalized to O(1))
w_track = 1.0
w_effort = 1e-4


# =============================================================================
# Model
# =============================================================================

print("="*60)
print("Nodal-Lefty Pattern Control - Centralized Training")
print("="*60)

model = SchnakenbergControlNet(
    features=(32, 64, 64),
    u_max=u_max,
    v_max=u_max,
    vel_max=V_max,
    L_domain=L_domain
)

key = jax.random.PRNGKey(42)
key, init_key = jax.random.split(key)

dummy_yn = jnp.zeros((N_grid, N_grid))
dummy_yl = jnp.zeros((N_grid, N_grid))
dummy_xi = jnp.zeros((n_agents, 2))

params = model.init(init_key, dummy_yn, dummy_yl, dummy_yn, dummy_yl, dummy_xi)
print(f"\nModel: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")


# =============================================================================
# Optimizer
# =============================================================================

# Lower learning rate for stability
lr_schedule = optax.exponential_decay(
    init_value=3e-4,
    transition_steps=3000,
    decay_rate=0.5
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr_schedule)
)
opt_state = optimizer.init(params)


# =============================================================================
# Loss & Training
# =============================================================================

def loss_fn(params, yn_init, yl_init, xi_init, yn_target, yl_target):
    yn_final, yl_final, xi_final, l_track, l_effort = unroll_with_loss(
        yn_init, yl_init, xi_init,
        yn_target, yl_target,
        params,
        model.apply,
        t_steps=T_steps,
        N_grid=N_grid,
        L=L_domain,
        dt=dt,
        D_n=D_n,
        D_l=D_l,
        gamma_n=gamma_n,
        gamma_l=gamma_l,
        n_n=n_n,
        n_l=n_l,
        k_n=k_n,
        k_l=k_l,
        alpha_n=alpha_n_initial,  # Use INITIAL alpha (striped) - controller must fight this!
        alpha_l=alpha_l_initial,
        beta_n=beta_n,
        beta_l=beta_l,
        sigma=sigma,
        u_max=u_max,
        V_max=V_max
    )
    return w_track * l_track + w_effort * l_effort, (l_track, l_effort)

batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0))

@partial(jax.jit)
def train_step(params, opt_state, yn_init, yl_init, xi_init, yn_target, yl_target):
    def mean_loss(p):
        losses, aux = batched_loss_fn(p, yn_init, yl_init, xi_init, yn_target, yl_target)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, aux)
    
    (loss, aux), grads = jax.value_and_grad(mean_loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux


# =============================================================================
# Main Loop
# =============================================================================

train_data = np.load(data_dir / 'train_data.npz')
pool_size = len(train_data['yn_init'])
print(f"Loaded {pool_size} training samples")

xi_template = jnp.stack(jnp.meshgrid(
    jnp.linspace(L_domain/10, L_domain*0.9, int(np.sqrt(n_agents))),
    jnp.linspace(L_domain/10, L_domain*0.9, int(np.sqrt(n_agents)))
), axis=-1).reshape(-1, 2)

metrics = []
start_time = time.time()

print(f"\nTraining with: T={T_steps}, Batch={batch_size}, LR=1e-4")

for epoch in trange(epochs):
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (batch_size,), 0, pool_size)
    
    params, opt_state, loss, aux = train_step(
        params, opt_state,
        jnp.array(train_data['yn_init'][idx]),
        jnp.array(train_data['yl_init'][idx]),
        jnp.tile(xi_template[None], (batch_size, 1, 1)),
        jnp.array(train_data['yn_target'][idx]),
        jnp.array(train_data['yl_target'][idx])
    )
    
    if epoch % 10 == 0:
        l_track, l_effort = aux
        metrics.append((epoch, float(loss), float(l_track), float(l_effort)))
        if epoch % 50 == 0:
            tqdm.write(f"Ep {epoch} | Loss: {loss:.4f} | Track: {l_track:.1f} | Effort: {l_effort:.4f}")

# Save
with open(Path(__file__).parent / 'nodal_lefty_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes(params))

# Plot
metrics = np.array(metrics)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(metrics[:,0], metrics[:,1]); axes[0].set_title('Total Loss'); axes[0].set_yscale('log')
axes[1].plot(metrics[:,0], metrics[:,2]); axes[1].set_title('Tracking Error'); axes[1].set_yscale('log')
plt.savefig(Path(__file__).parent / 'training_curves.png')

# =============================================================================
# Visualization of Test Cases
# =============================================================================

print("\n--- Generating Test Visualizations ---")
test_data = np.load(data_dir / 'test_data.npz')

# Pick 2 samples
idxs = [0, 1]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, idx in enumerate(idxs):
    yn_init = test_data['yn_init'][idx]
    yl_init = test_data['yl_init'][idx]
    yn_target = test_data['yn_target'][idx]
    yl_target = test_data['yl_target'][idx]
    
    # Run inference
    trajectory = unroll_jit(
        yn_init, yl_init, jnp.array(dummy_xi),  # dummy xi for now if needed or load from test
        yn_target, yl_target,
        params,
        model.apply,
        t_steps=T_steps,
        N_grid=N_grid,
        L=L_domain,
        dt=dt,
        D_n=D_n,
        D_l=D_l,
        gamma_n=gamma_n,
        gamma_l=gamma_l,
        n_n=n_n,
        n_l=n_l,
        k_n=k_n,
        k_l=k_l,
        alpha_n=alpha_n_initial,  # Use INITIAL alpha for consistency with training
        alpha_l=alpha_l_initial,
        beta_n=beta_n,
        beta_l=beta_l,
        sigma=sigma,
        u_max=u_max,
        V_max=V_max
    )
    yn_traj, yl_traj, _, _, _, _ = trajectory
    yn_final = yn_traj[-1]
    yl_final = yl_traj[-1]
    
    # Plot Column 1: Initial
    col_offset = i * 2
    
    # Row 0: Nodal
    im = axes[0, col_offset].imshow(yn_init.T, origin='lower', cmap='jet')
    axes[0, col_offset].set_title(f"Sample {idx}: Initial Nodal")
    plt.colorbar(im, ax=axes[0, col_offset])
    
    im = axes[0, col_offset+1].imshow(yl_init.T, origin='lower', cmap='jet')
    axes[0, col_offset+1].set_title(f"Sample {idx}: Initial Lefty")
    plt.colorbar(im, ax=axes[0, col_offset+1])

    # Row 1: Target
    im = axes[1, col_offset].imshow(yn_target.T, origin='lower', cmap='jet')
    axes[1, col_offset].set_title("Target Nodal")
    plt.colorbar(im, ax=axes[1, col_offset])
    
    im = axes[1, col_offset+1].imshow(yl_target.T, origin='lower', cmap='jet')
    axes[1, col_offset+1].set_title("Target Lefty")
    plt.colorbar(im, ax=axes[1, col_offset+1])

    # Row 2: Final (Controlled)
    im = axes[2, col_offset].imshow(yn_final.T, origin='lower', cmap='jet')
    axes[2, col_offset].set_title("Controlled Nodal")
    plt.colorbar(im, ax=axes[2, col_offset])
    
    im = axes[2, col_offset+1].imshow(yl_final.T, origin='lower', cmap='jet')
    axes[2, col_offset+1].set_title("Controlled Lefty")
    plt.colorbar(im, ax=axes[2, col_offset+1])

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'test_results.png')
print("Saved test_results.png")

print("Done!")
