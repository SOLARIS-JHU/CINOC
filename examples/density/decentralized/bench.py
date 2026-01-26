"""
Evaluation Script: Decentralized NS2D Shape Formation
Compares the trained Neural Policy against a Zero-Control baseline.
Metric: Mean Squared Error (MSE) between Final Density and Target Density.
"""

import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import flax.serialization
import jax
import jax.numpy as jnp
from functools import partial

# Prevent JAX from preallocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- Path Setup ---
# Assumes this script is in examples/ns2d/decentralized/
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.density.decentralized.dynamics import unroll_controlled
from models.policy_ns2d import DecentralizedNS2DControlNet

# --- 1. Configuration (Must Match Training) ---
N_AGENTS = 9         # 3x3 grid
T_STEPS = 150        # Horizon
N_EVAL = 32          # Number of evaluation samples

# Physics (Fan-Only Mode)
BUOYANCY = 0.0
SIGMA_PUSH = 0.2
PUSH_MAX = 0.8
PATCH_SIZE = 12
FEATURES = (32, 32, 64)

# --- 2. Helper: Zero Policy (Baseline) ---
def zero_policy_apply(params, rho, rho_target, xi):
    """
    Dummy policy: Fans are off (Zero Velocity).
    Returns: (u_force_x, u_force_y) for each agent.
    """
    n_batch = xi.shape[0]
    # Return zero velocity vector for each agent
    return jnp.zeros((n_batch, 2))

# --- 3. Data Loading ---
print("Loading configuration and data...")
data_dir = Path(__file__).parent.parent / 'data'
if not data_dir.exists():
    print(f"Error: Data directory not found at {data_dir}")
    sys.exit(1)

config = np.load(data_dir / 'config.npz')
Nx = int(config['Nx'])
Ny = int(config['Ny'])
dt = float(config['dt'])
print(f"Grid: {Nx}x{Ny}, dt: {dt}")

train_data = np.load(data_dir / 'train_data.npz')
pool_size = len(train_data['rho_init'])

# Select Random Validation Batch
key = jax.random.PRNGKey(101) # Validation Seed
idx = jax.random.randint(key, (N_EVAL,), 0, pool_size)

rho_init_batch = jnp.array(train_data['rho_init'][idx])
rho_target_batch = jnp.array(train_data['rho_target'][idx])

# Initialize Agents (3x3 Grid covering domain)
n_side = int(np.sqrt(N_AGENTS))
# Logic from training script to match agent placement
xi_template = jnp.stack(jnp.meshgrid(
    jnp.linspace(0.15, 0.85, n_side),
    jnp.linspace(0.15, 1.0, n_side)
), axis=-1).reshape(-1, 2)
xi_init_batch = jnp.tile(xi_template[None], (N_EVAL, 1, 1))

# --- 4. Load Model ---
model = DecentralizedNS2DControlNet(
    features=FEATURES,
    v_max=PUSH_MAX,
    patch_size=PATCH_SIZE
)

print("Loading trained parameters...")
param_file = Path(__file__).parent / 'ns2d_decentralized_params.msgpack'
try:
    with open(param_file, 'rb') as f:
        serialized_bytes = f.read()
except FileNotFoundError:
    print(f"Error: '{param_file}' not found. Run training script first.")
    sys.exit(1)

# Dummy init for structure
dummy_key = jax.random.PRNGKey(0)
dummy_rho = jnp.zeros((Nx, Ny))
dummy_xi = jnp.zeros((N_AGENTS, 2))
init_params = model.init(dummy_key, dummy_rho, dummy_rho, dummy_xi)

params = flax.serialization.from_bytes(init_params, serialized_bytes)

# --- 5. Evaluation Loop ---
print("Running simulations (Controlled vs Uncontrolled)...")

@jax.jit
def run_comparison(rho_init, xi_init, rho_target):
    # Physics args matching training
    # REMOVED 'n_agents' from this dictionary to fix the TypeError
    sim_args = {
        'Nx': Nx, 'Ny': Ny, 
        'dt': dt, 
        'buoyancy': BUOYANCY, 
        'sigma_push': SIGMA_PUSH, 
        'push_max': PUSH_MAX
    }
    
    # Controlled Run
    # unroll_controlled returns: (rho_traj, xi_traj, force_traj, ...)
    # Note: adjusting return unpacking based on standard dynamics return signature
    # If unroll_controlled returns more values (like loss terms), we capture only the first 3
    ret_c = unroll_controlled(
        rho_init, xi_init, rho_target, params, model.apply, T_STEPS, **sim_args
    )
    rho_c, xi_c = ret_c[0], ret_c[1]
    
    # Uncontrolled Run
    ret_u = unroll_controlled(
        rho_init, xi_init, rho_target, params, zero_policy_apply, T_STEPS, **sim_args
    )
    rho_u, xi_u = ret_u[0], ret_u[1]
    
    return (rho_c, xi_c), (rho_u, xi_u)

# Batch Execution
(traj_ctrl, traj_unc) = jax.vmap(run_comparison)(rho_init_batch, xi_init_batch, rho_target_batch)

rho_ctrl_all, xi_ctrl_all = traj_ctrl
rho_unc_all, xi_unc_all = traj_unc

# --- 6. Metrics (MSE) ---
print("Calculating metrics...")

# Target shape: (N, Nx, Ny) -> Expand for broadcasting if needed, 
# but rho_ctrl_all is (N, T, Nx, Ny). We compare Final State (-1).
mse_ctrl = jnp.mean((rho_ctrl_all[:, -1] - rho_target_batch)**2, axis=(1, 2))
mse_unc = jnp.mean((rho_unc_all[:, -1] - rho_target_batch)**2, axis=(1, 2))

print(f"Mean MSE (Controlled):   {jnp.mean(mse_ctrl):.6f}")
print(f"Mean MSE (Uncontrolled): {jnp.mean(mse_unc):.6f}")

# --- 7. Plotting ---
plt.figure(figsize=(16, 10))

# 1. Error Distribution
plt.subplot(2, 2, 1)
plt.boxplot([mse_ctrl, mse_unc], labels=['Controlled', 'Uncontrolled'])
plt.title(f'Shape Formation Error (MSE) (N={N_EVAL})')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 2. Visual Snapshot (Sample 0)
sample_idx = 0

# Target
plt.subplot(2, 3, 4)
plt.imshow(rho_target_batch[sample_idx].T, origin='lower', cmap='Blues', extent=[0,1,0,1])
plt.title('Target Shape')
plt.axis('off')

# Controlled Final
plt.subplot(2, 3, 5)
plt.imshow(rho_ctrl_all[sample_idx, -1].T, origin='lower', cmap='Blues', extent=[0,1,0,1])
# Overlay Agents
cx = xi_ctrl_all[sample_idx, -1, :, 0]
cy = xi_ctrl_all[sample_idx, -1, :, 1]
plt.scatter(cx, cy, c='orange', s=40, edgecolors='black', label='Fans')
# Draw velocity arrows (optional, taking last step velocity approx)
# For simplicity, just showing position
plt.title(f'Controlled Final (MSE={mse_ctrl[sample_idx]:.4f})')
plt.axis('off')

# Uncontrolled Final
plt.subplot(2, 3, 6)
plt.imshow(rho_unc_all[sample_idx, -1].T, origin='lower', cmap='Blues', extent=[0,1,0,1])
ux = xi_unc_all[sample_idx, -1, :, 0]
uy = xi_unc_all[sample_idx, -1, :, 1]
plt.scatter(ux, uy, c='gray', s=40, edgecolors='black', alpha=0.5)
plt.title(f'Uncontrolled Final (MSE={mse_unc[sample_idx]:.4f})')
plt.axis('off')

# 3. Agent Trajectories (Controlled)
plt.subplot(2, 2, 2)
# Plot background target for context
plt.imshow(rho_target_batch[sample_idx].T, origin='lower', cmap='Blues', alpha=0.3, extent=[0,1,0,1])
for i in range(N_AGENTS):
    plt.plot(xi_ctrl_all[sample_idx, :, i, 0], xi_ctrl_all[sample_idx, :, i, 1], alpha=0.7)
    plt.scatter(xi_ctrl_all[sample_idx, 0, i, 0], xi_ctrl_all[sample_idx, 0, i, 1], marker='x', c='k', s=10) # Start
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Fan Trajectories (Controlled)')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
save_path = Path(__file__).parent / 'ns2d_comparison_results.png'
plt.savefig(save_path)
print(f"Comparison plot saved to {save_path}")