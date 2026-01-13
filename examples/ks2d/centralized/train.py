"""
Centralized Training Script for 2D Kuramoto-Sivashinsky (KS)
Trains a Neural Policy to stabilize chaotic 2D turbulence.
"""

import jax
import jax.numpy as jnp
import optax
import time
import pickle
import flax.serialization
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from tqdm import trange
import sys

jax.config.update("jax_enable_x64", True)

# --- Local Imports ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from data_utils import get_batch_initial_conditions
from models.policy_ks2d import KS2DControlNet 

# --- 1. Configuration ---
CONFIG = {
    'N_grid': 64,         
    'L_domain': 32.0,      
    'dt': 0.005,            
    
    # Training
    'n_agents': 16,        # 4x4 grid of actuators
    'T_steps': 100,        # Horizon
    'batch_size': 8,       # Smaller batch size due to 2D memory usage
    'epochs': 500,
    'pool_size': 10,      # Number of precomputed chaotic ICs
    
    # Files
    'ic_filename': 'ks2d_chaotic_ics_64.pkl',
    'model_save_name': 'ks2d_centralized_params.msgpack'
}

# --- 2. Data Management ---
def get_or_create_data(config):
    """
    Manages loading/generating chaotic Initial Conditions (ICs).
    Saves data to a 'data' folder in the PARENT directory.
    """
    # Define path: ../data/ relative to this script
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / config['ic_filename']

    if file_path.exists():
        print(f"[Data] Found existing ICs at: {file_path}")
        print(f"[Data] Loading...")
        with open(file_path, 'rb') as f:
            u_pool = pickle.load(f)
        # Verify shape matches config
        if u_pool.shape[1] != config['N_grid']:
            print(f"[Data] WARNING: Saved data resolution {u_pool.shape[1]} != Config {config['N_grid']}.")
            print("[Data] Regenerating...")
        else:
            return jnp.array(u_pool)

    print(f"[Data] No valid ICs found. Generating {config['pool_size']} chaotic states...")
    print(f"[Data] This requires evolving physics to the attractor (i.e might take a while)...")
    
    # Use your data_utils function
    # Note: data_utils uses x64, so this generation is precise
    key = jax.random.PRNGKey(0)
    u_pool = get_batch_initial_conditions(
        key, 
        config['pool_size'], 
        config['N_grid'], 
        config['L_domain']
    )
    
    print(f"[Data] Saving to {file_path}...")
    # Save as numpy to avoid JAX versioning issues in pickle
    import numpy as np
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(u_pool), f)
        
    return u_pool

# --- 3. Loss Function ---
def loss_fn(params, u_init, xi_fixed, u_target, dynamics):
    """
    Computes trajectory loss for 2D KS.
    """
    # Unroll trajectory using the wrapper
    # u_traj shape: (T, N, N)
    u_traj, _, u_ctrl_traj, _ = dynamics.unroll_controlled(
        u_init, 
        xi_fixed, 
        u_target, 
        params, 
        t_steps=CONFIG['T_steps'],
        N_grid=CONFIG['N_grid'],
        L=CONFIG['L_domain'],
        dt=CONFIG['dt'],
        sigma=1.0 # Actuator width
    )
    
    # 1. Tracking Loss (Stabilize to Target)
    # Broadcast target to time dimension: (1, N, N) -> (T, N, N)
    traj_error = u_traj - u_target[None, :, :]
    l_track = jnp.mean(traj_error ** 2)
    
    # 2. Control Effort Loss
    # u_ctrl_traj shape: (T, M)
    l_effort = jnp.mean(u_ctrl_traj ** 2)
    
    # Total Loss
    # KS is chaotic and hard to control; high penalty on tracking
    total_loss = 100.0 * l_track + 1e-4 * l_effort
    
    return total_loss, (l_track, l_effort)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, u_init_batch, xi_fixed_batch, u_target_batch, dynamics):
    # Vectorize loss over the batch
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))
    
    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, u_init_batch, xi_fixed_batch, u_target_batch, dynamics)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 4. Main Training Script ---
if __name__ == "__main__":
    # A. Setup
    print("--- 2D KS Centralized Control Training ---")
    key = jax.random.PRNGKey(42)
    
    # Load Data
    u_init_pool = get_or_create_data(CONFIG)
    u_target_pool = jnp.zeros_like(u_init_pool) # Stabilize to zero
    
    # Setup Actuators (Fixed Grid)
    # 16 agents -> 4x4 grid
    grid_dim = int(jnp.sqrt(CONFIG['n_agents']))
    x_lin = jnp.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = jnp.meshgrid(x_lin, x_lin)
    xi_fixed_single = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
    
    # Batch the actuator positions
    xi_fixed_batch = jnp.tile(xi_fixed_single, (CONFIG['batch_size'], 1, 1))

    # Initialize Model & Optimizer
    model = KS2DControlNet(
        features=(32, 64), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=2.0
    )
    
    key, init_key = jax.random.split(key)
    dummy_u = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    params = model.init(init_key, dummy_u, dummy_u, dummy_xi)
    
    # Scheduler: warm start then decay
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=1e-3, warmup_steps=50, 
        decay_steps=CONFIG['epochs'], end_value=1e-5
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule)
    )
    opt_state = optimizer.init(params)
    
    # Initialize Dynamics Wrapper
    dynamics = PDEDynamics2D(policy_apply_fn=model.apply)

    # B. Training Loop
    metrics = []
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in trange(CONFIG['epochs']):
        key, subkey = jax.random.split(key)
        
        # Sample Batch
        idx = jax.random.randint(subkey, (CONFIG['batch_size'],), 0, CONFIG['pool_size'])
        u_init_b = u_init_pool[idx]
        u_target_b = u_target_pool[idx]
        
        # Step
        params, opt_state, loss, aux = train_step(
            params, opt_state, u_init_b, xi_fixed_batch, u_target_b, dynamics
        )
        
        track_loss, effort_loss = aux
        metrics.append([loss, track_loss, effort_loss])
        
        if epoch % 10 == 0:
            print(f"Ep {epoch} | Loss: {loss:.4f} | Track: {track_loss:.4f} | Effort: {effort_loss:.4f}")

    print(f"Training Complete. Time: {time.time()-start_time:.1f}s")
    
    # C. Save Results
    metrics = jnp.array(metrics)
    
    # Save Params
    with open(CONFIG['model_save_name'], 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Model saved to {CONFIG['model_save_name']}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(metrics[:, 1], label='Tracking MSE', color='blue')
    plt.plot(metrics[:, 2], label='Control Effort', color='orange', alpha=0.7)
    plt.yscale('log')
    plt.title(f'2D KS Control (L={CONFIG["L_domain"]}, N={CONFIG["N_grid"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('ks2d_training_metrics.png')
    print("Metrics plotted to ks2d_training_metrics.png")