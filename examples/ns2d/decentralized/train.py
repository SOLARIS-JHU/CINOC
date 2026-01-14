"""
Decentralized Training for NS2D Shape Formation Control

Train a decentralized policy network where each agent observes only
a local patch around its position. Exact replica of centralized training
but uses DecentralizedNS2DControlNet instead.
"""

import sys
from pathlib import Path
import os

# Prevent JAX from preallocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

from examples.ns2d.decentralized.dynamics import unroll_with_full_loss, unroll_controlled
from models.policy_ns2d import DecentralizedNS2DControlNet


# =============================================================================
# Hyperparameters (IDENTICAL to centralized, except policy)
# =============================================================================

# Grid/physics from config (loaded at runtime)
# These are set as module constants for sharing with visualize.py
N_AGENTS = 9         # 3x3 grid of stationary agents
T_STEPS = 150         # Simulation horizon
BATCH_SIZE = 4        # Reduced for memory
EPOCHS = 1000         # Train longer for decentralized

# Physics parameters (fan-only mode)
BUOYANCY = 0.0        # NO buoyancy - smoke only moves when pushed by fans
SIGMA_PUSH = 0.2      # Wide push influence

# Control limits (fan-only: no injection, just push velocity)
PUSH_MAX = 0.8        # Max push velocity
FEATURES = (16, 32)   # CNN feature channels - keep same
PATCH_SIZE = 12      

# Loss weights - NEW OBJECTIVE DESIGN
# W_TRANSPORT = 10.0   #20# PRIMARY: push smoke centroid → target centroid
# W_HOLD = 50.0         #10# TERMINAL: smoke must stay at target (time-weighted)
# W_FIND = 10.0         # EXPLORATION: agents approach smoke
# W_SURROUND = 10.0     #15# Agents form ring around target
# W_BRAKE = 40.0        # Slow down when at target
# W_COLL = 1500.0        # Avoid other agents
# W_BOUND = 20.0        # Stay in domain
# W_SMOOTH = 0.1        # Velocity smoothness
# W_EFFORT = 0.001      # Control energy
# W_MASS = 80.0         #50# NEW: Preserve smoke mass
# R_SAFE = 0.15         # Safe radius for collision

W_TRANSPORT = 0.0   #20# PRIMARY: push smoke centroid → target centroid
W_HOLD = 10.0 #0.3        #10# TERMINAL: smoke must stay at target (time-weighted)
W_FIND = 0.0         # EXPLORATION: agents approach smoke
W_SURROUND = 0.0     #15# Agents form ring around target
W_BRAKE = 0.0        # Slow down when at target
W_COLL = 70.0        # Avoid other agents
W_BOUND = 10.0        # Stay in domain
W_SMOOTH = 0.1        # Velocity smoothness
W_EFFORT = 0.001      # Control energy
W_MASS = 30.0         #50# NEW: Preserve smoke mass
R_SAFE = 0.15         # Safe radius for collision


# =============================================================================
# Training
# =============================================================================

def main():
    print("="*60)
    print("NS2D Shape Formation - Decentralized Training")
    print("="*60)
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data'
    if not data_dir.exists():
        print(f"Data not found at {data_dir}")
        print("Run: python examples/ns2d/generate_dataset.py")
        return
    
    config = np.load(data_dir / 'config.npz')
    Nx = int(config['Nx'])
    Ny = int(config['Ny'])
    dt = float(config['dt'])
    
    # Use module-level constants
    n_agents = N_AGENTS
    buoyancy = BUOYANCY
    sigma_push = SIGMA_PUSH
    
    print(f"\nGrid: {Nx}x{Ny}, Agents: {n_agents} (decentralized)")
    
    train_data = np.load(data_dir / 'train_data.npz')
    pool_size = len(train_data['rho_init'])
    print(f"Training samples: {pool_size}")
    
    # Use module-level hyperparameters
    T_steps = T_STEPS
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    push_max = PUSH_MAX
    
    # Model (Decentralized - local patch observation)
    model = DecentralizedNS2DControlNet(
        features=FEATURES,
        v_max=push_max,
        patch_size=PATCH_SIZE
    )
    
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    dummy_smoke = jnp.zeros((Nx, Ny))
    dummy_xi = jnp.zeros((n_agents, 2))
    params = model.init(init_key, dummy_smoke, dummy_smoke, dummy_xi)
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    lr_schedule = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=2000,
        decay_rate=0.5
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule)
    )
    opt_state = optimizer.init(params)
    
    # Loss function - with mass conservation
    def loss_fn(params, smoke_init, xi_init, rho_target):
        smoke_final, xi_final, l_transport, l_hold, l_find, l_surround, l_brake, l_coll, l_bound, l_smooth, l_effort, l_mass = unroll_with_full_loss(
            smoke_init, xi_init, rho_target, params, model.apply, T_steps,
            Nx=Nx, Ny=Ny, n_agents=n_agents, dt=dt, buoyancy=buoyancy,
            sigma_push=sigma_push, push_max=push_max, R_safe=R_SAFE
        )
        
        total_loss = W_TRANSPORT * l_transport + W_HOLD * l_hold + W_FIND * l_find + \
                     W_SURROUND * l_surround + W_BRAKE * l_brake + W_MASS * l_mass + \
                     W_COLL * l_coll + W_BOUND * l_bound + W_SMOOTH * l_smooth + W_EFFORT * l_effort
        
        return total_loss, (l_transport, l_hold, l_find, l_surround, l_brake, l_coll, l_bound, l_smooth, l_effort, l_mass)
    
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))
    
    @jax.jit
    def train_step(params, opt_state, smoke_init, xi_init, rho_target):
        def mean_loss(p):
            losses, aux = batched_loss_fn(p, smoke_init, xi_init, rho_target)
            return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, aux)
        
        (loss, aux), grads = jax.value_and_grad(mean_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux
    
    # Initial agent positions (3x3 grid covering domain)
    n_side = int(np.sqrt(n_agents))
    xi_template = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.15, 0.85, n_side),  # Cover x
        jnp.linspace(0.15, 1.0, n_side)     # Cover y (full height)
    ), axis=-1).reshape(-1, 2)
    
    # Training loop
    metrics = []
    start_time = time.time()
    
    print(f"\nTraining: T={T_steps}, Batch={batch_size}, Epochs={epochs}")
    
    for epoch in trange(epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, pool_size)
        
        smoke_batch = jnp.array(train_data['rho_init'][idx])
        target_batch = jnp.array(train_data['rho_target'][idx])
        xi_batch = jnp.tile(xi_template[None], (batch_size, 1, 1))
        
        params, opt_state, loss, aux = train_step(
            params, opt_state, smoke_batch, xi_batch, target_batch
        )
        
        if epoch % 10 == 0:
            l_transport, l_hold, l_find, l_surround, l_brake, l_coll, l_bound, l_smooth, l_effort, l_mass = aux
            metrics.append((epoch, float(loss), float(l_transport), float(l_hold), float(l_find), 
                          float(l_surround), float(l_brake), float(l_coll), float(l_bound), 
                          float(l_smooth), float(l_effort), float(l_mass)))
            
            if epoch % 50 == 0:
                tqdm.write(f"Ep {epoch:4d} | Loss: {loss:.3f} | Trans: {l_transport:.4f} | Hold: {l_hold:.4f} | Find: {l_find:.4f}")
                tqdm.write(f"         | Surr: {l_surround:.4f} | Brake: {l_brake:.4f} | Coll: {l_coll:.6f} | Bound: {l_bound:.6f}")
                tqdm.write(f"         | Smooth: {l_smooth:.6f} | Effort: {l_effort:.6f} | Mass: {l_mass:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    
    # Save
    save_path = Path(__file__).parent / 'ns2d_decentralized_params.msgpack'
    with open(save_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Saved: {save_path}")
    
    # Plot training curves (3 panels)
    metrics = np.array(metrics)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(metrics[:, 0], metrics[:, 1])
    axes[0].set_title('Total Loss')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    
    axes[1].plot(metrics[:, 0], metrics[:, 2])
    axes[1].set_title('Tracking Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    
    axes[2].plot(metrics[:, 0], metrics[:, 3])
    axes[2].set_title('Effort Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'training_curves.png', dpi=150)
    print("Saved: training_curves.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
