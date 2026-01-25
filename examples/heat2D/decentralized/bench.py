"""
Evaluation Script: Controlled vs Uncontrolled Performance (Heat 2D)
Compares the trained Decentralized Heat2D Policy against a zero-control baseline.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
import sys
import argparse
from pathlib import Path
from functools import partial
from tesseract_core import Tesseract

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
from data_utils import get_training_data

# --- 1. Setup & Configuration ---
# NOTE: Using the Heat 2D solver image
solver_ts = Tesseract.from_image("solver_heat2d_decentralized:latest")

# Config must match training script
n_grid = 32
n_agents = 16
T_steps = 300
N_eval = 100  # Number of evaluation samples

model = DecentralizedHeat2DControlNet(features=(16, 32))

# --- 2. Helper: Zero Policy for Uncontrolled Baseline ---
def zero_policy_apply(params, local_z, z_target, local_xi):
    """
    Dummy policy that returns zero control inputs for Heat 2D.
    Must accept 4 arguments to match solver interface:
    (params, z_observed, z_target, xi_curr)
    """
    n_batch = local_xi.shape[0]
    # Return zeros for forcing (u) and velocity (v)
    # u shape: (batch_size,) -> scalar intensity per agent
    # v shape: (batch_size, 2) -> vector velocity per agent
    return jnp.zeros((n_batch,)), jnp.zeros((n_batch, 2))

# --- 3. Data Generation & Loading ---
print(f"Loading/Generating {N_eval} Evaluation Samples...")

# We use the same data utility but force it to generate new data if needed
# or rely on the randomness of the selection if loading a large dataset.
# Ideally, we should generate fresh data for pure validation.
# For simplicity here, we assume the get_training_data can return enough samples
# and we pick a random subset with a validation seed.

# Load a potentially large dataset
z_init_pool, z_target_pool, _ = get_training_data(
    n_samples=2000, # Load enough to pick random validation set
    n_grid=n_grid,
    dataset_dir='../data'
)

# Pick N_eval random indices using a validation seed
val_key = jax.random.PRNGKey(4242)
idx = jax.random.randint(val_key, (N_eval,), 0, 2000)
z_init_batch = z_init_pool[idx]
z_target_batch = z_target_pool[idx]

# Initialize Agents (Grid Pattern)
n_side = int(jnp.sqrt(n_agents))
spacing = 0.8 / (n_side + 1)
xi_template = []
for i in range(n_side):
    for j in range(n_side):
        if len(xi_template) < n_agents:
            xi_template.append([0.1 + spacing * (i+1), 0.1 + spacing * (j+1)])
xi_init_single = jnp.array(xi_template)
xi_init_batch = jnp.tile(xi_init_single, (N_eval, 1, 1))

# Load Trained Parameters
print("Loading trained parameters...")
try:
    with open('decentralized_params_heat2d.msgpack', 'rb') as f:
        serialized_bytes = f.read()
except FileNotFoundError:
    print("Error: 'decentralized_params_heat2d.msgpack' not found. Run training first.")
    sys.exit(1)

# Initialize dummy params
dummy_key = jax.random.PRNGKey(0)
dummy_z = jnp.zeros((n_grid, n_grid))
dummy_xi = jnp.zeros((n_agents, 2))
dummy_params = model.init(dummy_key, dummy_z, dummy_z, dummy_xi)

params = flax.serialization.from_bytes(dummy_params, serialized_bytes)

# --- 4. Evaluation Loop ---
with solver_ts:
    # A. Controlled Dynamics
    dynamics_ctrl = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)
    
    # B. Uncontrolled Dynamics
    dynamics_unc = PDEDynamics(solver_ts, policy_apply_fn=zero_policy_apply, use_tesseract=False)

    print("Running simulations (this may take a moment)...")

    def run_comparison(z_init, xi_init, z_target):
        # Controlled run
        z_c, xi_c, _, _ = dynamics_ctrl.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        # Uncontrolled run
        z_u, xi_u, _, _ = dynamics_unc.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        return (z_c, xi_c), (z_u, xi_u)

    # Vmap over the batch
    # Note: 2D PDE trajectories are heavy (N_eval, T_steps, N_grid, N_grid). 
    # If OOM, reduce N_eval or process in chunks.
    (traj_ctrl, traj_unc) = jax.vmap(run_comparison)(z_init_batch, xi_init_batch, z_target_batch)
    
    # Unpack results
    z_ctrl_all, xi_ctrl_all = traj_ctrl
    z_unc_all, xi_unc_all = traj_unc

# --- 5. Analysis & Visualization ---
print("Calculating metrics...")

# Calculate MSE against Target
# Target shape: (N_eval, N_grid, N_grid) -> Expand to (N_eval, 1, N_grid, N_grid)
targets_expanded = z_target_batch[:, None, :, :]

mse_ctrl = jnp.mean((z_ctrl_all - targets_expanded)**2, axis=(1, 2, 3))
mse_unc = jnp.mean((z_unc_all - targets_expanded)**2, axis=(1, 2, 3))

print(f"Average MSE (Controlled):   {jnp.mean(mse_ctrl):.6f}")
print(f"Average MSE (Uncontrolled): {jnp.mean(mse_unc):.6f}")

# --- Plotting ---
plt.figure(figsize=(16, 10))

# 1. Error Distribution
plt.subplot(2, 2, 1)
plt.boxplot([mse_ctrl, mse_unc], labels=['Controlled', 'Uncontrolled'])
plt.title(f'Tracking MSE Distribution (N={N_eval})')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 2. Visual Snapshot (Controlled) - Final State
sample_idx = 0
vmin, vmax = 0, 1.0 # Assuming heat roughly in [0,1]

plt.subplot(2, 3, 4)
plt.imshow(z_target_batch[sample_idx], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title('Target State')

plt.subplot(2, 3, 5)
plt.imshow(z_ctrl_all[sample_idx, -1], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
plt.colorbar()
# Overlay agents
plt.scatter(xi_ctrl_all[sample_idx, -1, :, 0]*n_grid, xi_ctrl_all[sample_idx, -1, :, 1]*n_grid, 
            c='cyan', s=20, edgecolors='white', label='Agents')
plt.title(f'Controlled Final (MSE={mse_ctrl[sample_idx]:.4f})')

plt.subplot(2, 3, 6)
plt.imshow(z_unc_all[sample_idx, -1], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title(f'Uncontrolled Final (MSE={mse_unc[sample_idx]:.4f})')

# 3. Agent Trajectories (Controlled, Sample 0)
plt.subplot(2, 2, 2)
for i in range(n_agents):
    plt.plot(xi_ctrl_all[sample_idx, :, i, 0], xi_ctrl_all[sample_idx, :, i, 1], alpha=0.6)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Controlled Agent Trajectories (2D)')
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.savefig('heat2d_comparison_results.png')
print("Comparison plot saved to 'heat2d_comparison_results.png'")