"""
NS2D Shape Formation Zero-Shot Scalability Experiment.
Trains on N=16 agents (4x4), Evaluates on N=[4, 9, ..., 100].
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
import pandas as pd
import numpy as np
import flax.serialization
import matplotlib.pyplot as plt
from functools import partial
from tqdm import trange
from matplotlib.ticker import PercentFormatter

# --- Local Imports ---
from examples.density.decentralized.dynamics import unroll_with_full_loss, unroll_controlled
from models.policy_ns2d import DecentralizedNS2DControlNet

# --- 1. Configuration & Directories ---
SAVE_DIR = Path("figures/ns2d_zs_scaling")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# UPDATED: Filename for N=16 model
MODEL_NAME = "ns2d_policy_n16.msgpack" 
MODEL_PATH = SAVE_DIR / MODEL_NAME
CSV_PATH = SAVE_DIR / "ns2d_zs_results_n16.csv"
PLOT_PATH = SAVE_DIR / "ns2d_zs_relative_mse_n16.pdf"

CONFIG = {
    'N_AGENTS_TRAIN': 16,      # UPDATED: Training Baseline (4x4)
    'T_STEPS': 150,
    'BATCH_SIZE': 4,
    'EPOCHS': 1000,           
    'BUOYANCY': 0.0,
    'SIGMA_PUSH': 0.2,
    'PUSH_MAX': 0.8,
    'FEATURES': (32, 32, 64),
    'PATCH_SIZE': 12,
    'R_SAFE': 0.15,
    # Loss Weights
    'W_HOLD': 3.0,
    'W_COLL': 10.0,
    'W_BOUND': 10.0,
    'W_SMOOTH': 0.1,
    'W_EFFORT': 0.001,
    'W_MASS': 5.0
}

# --- 2. Helper: Agent Grid Initializer ---
def get_grid_xi(n_agents):
    """Generates a regular grid of initial positions in [0.15, 0.85]."""
    n_side = int(jnp.ceil(jnp.sqrt(n_agents)))
    x = jnp.linspace(0.15, 0.85, n_side)
    y = jnp.linspace(0.15, 0.85, n_side)
    xv, yv = jnp.meshgrid(x, y) # Default xy indexing
    xi = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
    return xi[:n_agents]

# --- 3. Training Step ---
@partial(jax.jit, static_argnames=('model_apply', 'optimizer', 'T_steps', 'Nx', 'Ny', 'n_agents', 'dt'))
def train_step(params, opt_state, smoke_init, xi_init, rho_target, 
               model_apply, optimizer, T_steps, Nx, Ny, n_agents, dt):
    
    def mean_loss_fn(p):
        losses = jax.vmap(unroll_with_full_loss, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None))(
            smoke_init, xi_init, rho_target, p, model_apply, 
            T_steps, Nx, Ny, n_agents, dt, 
            CONFIG['BUOYANCY'], CONFIG['SIGMA_PUSH'], CONFIG['PUSH_MAX'], CONFIG['R_SAFE']
        )
        _, _, l_hold, l_coll, l_bound, l_smooth, l_effort, l_mass = losses
        
        total_loss = (CONFIG['W_HOLD'] * l_hold + 
                      CONFIG['W_MASS'] * l_mass + 
                      CONFIG['W_COLL'] * l_coll + 
                      CONFIG['W_BOUND'] * l_bound + 
                      CONFIG['W_SMOOTH'] * l_smooth + 
                      CONFIG['W_EFFORT'] * l_effort)
        
        return jnp.mean(total_loss), (jnp.mean(l_hold), jnp.mean(l_effort))

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 4. Main Execution ---
def main():
    print(f"--- NS2D Shape Formation Zero-Shot Scaling (Train N={CONFIG['N_AGENTS_TRAIN']}) ---")
    
    # 1. Load Data
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    if not data_dir.exists():
        # Fallback for standalone runs
        print(f"Data directory {data_dir} not found. Using dummy data.")
        Nx, Ny, dt = 64, 80, 1.0
        pool_size = 100
        rho_init_pool = jnp.zeros((pool_size, Nx, Ny))
        rho_target_pool = jnp.zeros((pool_size, Nx, Ny))
    else:
        config_file = np.load(data_dir / 'config.npz')
        Nx = int(config_file['Nx'])
        Ny = int(config_file['Ny'])
        dt = float(config_file['dt'])
        
        train_data = np.load(data_dir / 'train_data.npz')
        rho_init_pool = jnp.array(train_data['rho_init'])
        rho_target_pool = jnp.array(train_data['rho_target'])
        
    pool_size = len(rho_init_pool)
    print(f"Grid: {Nx}x{Ny}, dt: {dt}, Data Size: {pool_size}")

    # 2. Setup Model
    model = DecentralizedNS2DControlNet(
        features=CONFIG['FEATURES'],
        v_max=CONFIG['PUSH_MAX'],
        patch_size=CONFIG['PATCH_SIZE']
    )
    
    # Init
    key = jax.random.PRNGKey(42)
    dummy_smoke = jnp.zeros((Nx, Ny))
    dummy_xi = get_grid_xi(CONFIG['N_AGENTS_TRAIN'])
    
    key, init_key = jax.random.split(key)
    init_params = model.init(init_key, dummy_smoke, dummy_smoke, dummy_xi)
    
    lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))

    # --- Training ---
    if not MODEL_PATH.exists():
        print(f"Training NS2D Policy on N={CONFIG['N_AGENTS_TRAIN']} agents...")
        params = init_params
        opt_state = optimizer.init(params)
        
        xi_train_batch = jnp.tile(dummy_xi, (CONFIG['BATCH_SIZE'], 1, 1))
        
        pbar = trange(CONFIG['EPOCHS'], desc="Training")
        for _ in pbar:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (CONFIG['BATCH_SIZE'],), 0, pool_size)
            
            params, opt_state, loss, aux = train_step(
                params, opt_state, 
                rho_init_pool[idx], xi_train_batch, rho_target_pool[idx],
                model.apply, optimizer,
                CONFIG['T_STEPS'], Nx, Ny, CONFIG['N_AGENTS_TRAIN'], dt
            )
            l_hold_val, l_eff_val = aux
            pbar.set_postfix({"Loss": f"{loss:.4f}", "Hold": f"{l_hold_val:.4f}"})
            
        with open(MODEL_PATH, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
        print("Model saved.")
    else:
        print(f"Loading NS2D model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            params = flax.serialization.from_bytes(init_params, f.read())

    # --- Zero-Shot Evaluation ---
    # UPDATED: Range of N to test (4, 9, 16(base), 25, 36, 49, 64, 81, 100)
    n_eval_list = [4, 9, 16, 25, 36, 49, 64, 81, 100]
    
    print(f"\nStarting Zero-Shot Sweep on N={n_eval_list}...")
    results = []
    
    n_test = 5
    rho_init_test = rho_init_pool[-n_test:]
    rho_target_test = rho_target_pool[-n_test:]
    
    for n in n_eval_list:
        xi_eval = get_grid_xi(n)
        mse_list = []
        effort_list = []
        
        for i in range(n_test):
            smoke_traj, _, v_traj = unroll_controlled(
                rho_init_test[i], 
                xi_eval, 
                rho_target_test[i], 
                params, 
                model.apply, 
                CONFIG['T_STEPS'], 
                Nx, 
                Ny, 
                dt, 
                CONFIG['BUOYANCY'], 
                CONFIG['SIGMA_PUSH'], 
                CONFIG['PUSH_MAX']
            )
            
            final_smoke = smoke_traj[-1]
            mse = float(jnp.mean((final_smoke - rho_target_test[i])**2))
            mse_list.append(mse)
            effort_list.append(float(jnp.mean(v_traj**2)))
        
        avg_mse = np.mean(mse_list)
        avg_effort = np.mean(effort_list)
        
        print(f"N={n:3d} | MSE: {avg_mse:.6f} | Effort: {avg_effort:.5f}")
        results.append({"n_agents": n, "mse": avg_mse, "effort": avg_effort})

    # --- Plotting ---
    df = pd.DataFrame(results)
    
    baseline_mse = df[df['n_agents'] == CONFIG['N_AGENTS_TRAIN']]['mse'].values[0]
    baseline_mse = max(baseline_mse, 1e-9)
    df['relative_mse'] = (df['mse'] / baseline_mse) * 100
    df.to_csv(CSV_PATH, index=False)
    
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    color = '#8e44ad'
    ax.plot(df['n_agents'], df['relative_mse'], marker='s', linestyle='-', 
            color=color, linewidth=2, markersize=8, label='Relative Shape Error')
    
    ax.axvline(x=CONFIG['N_AGENTS_TRAIN'], color='#e67e22', linestyle='--', alpha=0.8, 
               label=f'Training Size (N={CONFIG["N_AGENTS_TRAIN"]})')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)

    # UPDATED Title
    ax.set_title(f"NS2D Zero-Shot Scalability\n(Train: 4x4 Grid)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Number of Agents (N)", fontsize=10)
    ax.set_ylabel("Relative MSE (%)", color=color, fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(PLOT_PATH)
    print(f"\nAnalysis complete. Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()