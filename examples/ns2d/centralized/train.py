"""
Centralized Training for NS2D Shape Formation Control

Train a policy network to control movable smoke injectors to achieve
target smoke shapes.
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

from examples.ns2d.centralized.dynamics import unroll_with_full_loss, unroll_controlled
from models.policy_ns2d import NS2DControlNet


# =============================================================================
# Hyperparameters (can be imported by visualize.py)
# =============================================================================

# Grid/physics from config (loaded at runtime)
# These are set as module constants for sharing with visualize.py
N_AGENTS = 9         # 4x4 grid of stationary agents
T_STEPS = 100         # Simulation horizon
BATCH_SIZE = 2        # Reduced for memory
EPOCHS = 1000

# Physics parameters (conservative for fan control)
BUOYANCY = 0.0        # NO buoyancy - smoke only moves when pushed by fans
SIGMA_INJECT = 0.08   # Not used in fan-only mode
SIGMA_PUSH = 0.2      # Wide push influence

# Control limits
U_MAX = 1.0           # Max injection intensity
PUSH_MAX = 0.2        # Max push velocity (new!)
FEATURES = (16, 32)   # CNN feature channels

# Loss weights (tuned for blob transport)
# Primary objective: reach target position with correct shape
W_TRACK = 10.0        # Tracking loss - MAIN objective
W_SHAPE = 0.0         # Shape matching - DISABLED
# Constraints: prevent bad behavior
W_EFFORT = 0.001      # Effort regularization (keep controls reasonable)
W_BOUND = 20.0        # Boundary penalty (agents stay in domain)
W_COLL = 400.0         # Collision avoidance (agents don't overlap)
W_ACCEL = 0.05        # Acceleration smoothness (smooth control signals)
R_SAFE = 0.08         # Collision radius


# =============================================================================
# Training
# =============================================================================

def main():
    print("="*60)
    print("NS2D Shape Formation - Centralized Training")
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
    sigma_inject = SIGMA_INJECT
    sigma_push = SIGMA_PUSH
    
    print(f"\nGrid: {Nx}x{Ny}, Agents: {n_agents} (stationary)")
    
    train_data = np.load(data_dir / 'train_data.npz')
    pool_size = len(train_data['rho_init'])
    print(f"Training samples: {pool_size}")
    
    # Use module-level hyperparameters
    T_steps = T_STEPS
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    u_max = U_MAX
    push_max = PUSH_MAX
    
    # Model (Fan-only - agents push smoke, don't inject)
    model = NS2DControlNet(
        features=FEATURES,
        v_max=push_max  # Only push velocity matters
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
    
    # Loss function with velocity control and shape preservation
    def loss_fn(params, smoke_init, xi_init, rho_target):
        smoke_final, xi_final, l_track, l_effort, l_bound, l_coll, l_accel, l_shape = unroll_with_full_loss(
            smoke_init, xi_init, rho_target, params, model.apply, T_steps,
            Nx=Nx, Ny=Ny, n_agents=n_agents, dt=dt, buoyancy=buoyancy,
            sigma_inject=sigma_inject, sigma_push=sigma_push,
            u_max=u_max, push_max=push_max, R_safe=R_SAFE
        )
        
        total_loss = W_TRACK * l_track + W_EFFORT * l_effort + \
                     W_BOUND * l_bound + W_COLL * l_coll + W_ACCEL * l_accel + \
                     W_SHAPE * l_shape
        
        return total_loss, (l_track, l_effort, l_coll, l_shape)
    
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
    
    # Initial agent positions (5x5 grid covering domain)
    n_side = int(np.sqrt(n_agents))  # Should be 5 for 25 agents
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
            l_track, l_effort, l_coll, l_shape = aux
            metrics.append((epoch, float(loss), float(l_track), float(l_effort), 
                          float(l_coll), float(l_shape)))
            
            if epoch % 50 == 0:
                tqdm.write(f"Ep {epoch} | Loss: {loss:.4f} | Track: {l_track:.4f} | " +
                          f"Effort: {l_effort:.4f} | Coll: {l_coll:.4f} | Shape: {l_shape:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    
    # Save
    save_path = Path(__file__).parent / 'ns2d_params.msgpack'
    with open(save_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Saved: {save_path}")
    
    # Plot training curves (2x2 like heat2D)
    metrics = np.array(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(metrics[:, 0], metrics[:, 1])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].plot(metrics[:, 0], metrics[:, 2])
    axes[0, 1].set_title('Tracking Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].plot(metrics[:, 0], metrics[:, 3])
    axes[1, 0].set_title('Effort Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].plot(metrics[:, 0], metrics[:, 4], label='Collision')
    axes[1, 1].plot(metrics[:, 0], metrics[:, 5], label='Shape')
    axes[1, 1].set_title('Collision & Shape Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'training_curves.png', dpi=150)
    print("Saved: training_curves.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
