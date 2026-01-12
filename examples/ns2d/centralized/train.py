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

from examples.ns2d.centralized.dynamics import unroll_with_loss, unroll_controlled
from models.policy_ns2d import NS2DControlNet


# =============================================================================
# Hyperparameters (can be imported by visualize.py)
# =============================================================================

# Grid/physics from config (loaded at runtime)
# These are set as module constants for sharing with visualize.py
N_AGENTS = 25         # 5x5 grid of agents
T_STEPS = 100         # Simulation horizon
BATCH_SIZE = 4        # Reduced for CNN memory
EPOCHS = 500
U_MAX = 10.0          # Max injection intensity
V_MAX = 0.5           # Max agent velocity
SIGMA = 0.05          # Agent kernel width
W_TRACK = 1.0         # Tracking loss weight
W_EFFORT = 0.01       # Effort loss weight
FEATURES = (16, 32)   # CNN feature channels


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
    buoyancy = float(config['buoyancy'])
    
    # Use module-level constants (not from config)
    n_agents = N_AGENTS
    sigma = SIGMA
    
    print(f"\nGrid: {Nx}x{Ny}, Agents: {n_agents}")
    
    train_data = np.load(data_dir / 'train_data.npz')
    pool_size = len(train_data['rho_init'])
    print(f"Training samples: {pool_size}")
    
    # Use module-level hyperparameters
    T_steps = T_STEPS
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    u_max = U_MAX
    v_max = V_MAX
    w_track = W_TRACK
    w_effort = W_EFFORT
    
    # Model (CNN-based, no pooling)
    model = NS2DControlNet(
        features=FEATURES,
        u_max=u_max,
        v_max=v_max
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
    
    # Loss function
    def loss_fn(params, smoke_init, xi_init, rho_target):
        smoke_final, xi_final, l_track, l_effort = unroll_with_loss(
            smoke_init, xi_init, rho_target, params, model.apply, T_steps,
            Nx=Nx, Ny=Ny, dt=dt, buoyancy=buoyancy, sigma=sigma,
            u_max=u_max, v_max=v_max
        )
        return w_track * l_track + w_effort * l_effort, (l_track, l_effort)
    
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
            l_track, l_effort = aux
            metrics.append((epoch, float(loss), float(l_track), float(l_effort)))
            
            if epoch % 50 == 0:
                tqdm.write(f"Ep {epoch} | Loss: {loss:.4f} | Track: {l_track:.4f} | Effort: {l_effort:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    
    # Save
    save_path = Path(__file__).parent / 'ns2d_params.msgpack'
    with open(save_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Saved: {save_path}")
    
    # Plot training curves
    metrics = np.array(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(metrics[:, 0], metrics[:, 1])
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_yscale('log')
    
    axes[1].plot(metrics[:, 0], metrics[:, 2], label='Tracking')
    axes[1].plot(metrics[:, 0], metrics[:, 3], label='Effort')
    axes[1].set_title('Loss Components')
    axes[1].set_xlabel('Epoch')
    axes[1].set_yscale('log')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'training_curves.png', dpi=150)
    print("Saved: training_curves.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
