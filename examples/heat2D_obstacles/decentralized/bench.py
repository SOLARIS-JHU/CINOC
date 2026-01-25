"""
Evaluation Script: Heat 2D Control with Obstacles
Compares the trained Decentralized Policy against a zero-control baseline,
checking for tracking performance and obstacle avoidance.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
import sys
from pathlib import Path
from functools import partial
from tesseract_core import Tesseract

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
from data_utils import get_training_data

# --- 1. Configuration ---
# NOTE: Using the Heat 2D solver image
solver_ts = Tesseract.from_image("solver_heat2d_decentralized:latest")

# Must match training config
n_grid = 32
n_agents = 16
T_steps = 300
N_eval = 100
R_safe = 0.08 

# Obstacle Config (Matches training)
# [x, y, radius]
OBSTACLES = jnp.array([
    [0.15, 0.50, 0.08],   # Left middle
    [0.85, 0.50, 0.08],   # Right middle
    [0.50, 0.15, 0.08],   # Bottom middle
])

model = DecentralizedHeat2DControlNet(features=(16, 32))

# --- 2. Helper: Zero Policy ---
def zero_policy_apply(params, local_z, z_target, local_xi):
    """
    Dummy policy: returns zero forcing and zero velocity.
    """
    n_batch = local_xi.shape[0]
    return jnp.zeros((n_batch,)), jnp.zeros((n_batch, 2))

# --- 3. Data Generation & Loading ---
print(f"Loading/Generating {N_eval} Evaluation Samples...")

# Load dataset (using same utility as training)
z_init_pool, z_target_pool, _ = get_training_data(
    n_samples=2000, 
    n_grid=n_grid,
    dataset_dir='../../heat2D/data' # Adjusted path based on typical structure
)

# Pick random validation subset
val_key = jax.random.PRNGKey(4242)
idx = jax.random.randint(val_key, (N_eval,), 0, len(z_init_pool))
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

# Load Parameters
print("Loading trained parameters...")
param_file = 'decentralized_params_heat2d_obstacles.msgpack'
try:
    with open(param_file, 'rb') as f:
        serialized_bytes = f.read()
except FileNotFoundError:
    print(f"Error: '{param_file}' not found. Run training script first.")
    sys.exit(1)

# Restore params
dummy_key = jax.random.PRNGKey(0)
dummy_z = jnp.zeros((n_grid, n_grid))
dummy_xi = jnp.zeros((n_agents, 2))
dummy_params = model.init(dummy_key, dummy_z, dummy_z, dummy_xi)
params = flax.serialization.from_bytes(dummy_params, serialized_bytes)

# --- 4. Evaluation Loop ---
with solver_ts:
    # A. Controlled
    dynamics_ctrl = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)
    # B. Uncontrolled
    dynamics_unc = PDEDynamics(solver_ts, policy_apply_fn=zero_policy_apply, use_tesseract=False)

    print("Running simulations...")

    def run_comparison(z_init, xi_init, z_target):
        # Controlled
        z_c, xi_c, _, _ = dynamics_ctrl.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        # Uncontrolled
        z_u, xi_u, _, _ = dynamics_unc.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        return (z_c, xi_c), (z_u, xi_u)

    # Batch Processing
    (traj_ctrl, traj_unc) = jax.vmap(run_comparison)(z_init_batch, xi_init_batch, z_target_batch)
    
    z_ctrl_all, xi_ctrl_all = traj_ctrl
    z_unc_all, xi_unc_all = traj_unc

# --- 5. Analysis ---
print("Calculating metrics...")

# MSE Calculation
targets_expanded = z_target_batch[:, None, :, :]
mse_ctrl = jnp.mean((z_ctrl_all - targets_expanded)**2, axis=(1, 2, 3))
mse_unc = jnp.mean((z_unc_all - targets_expanded)**2, axis=(1, 2, 3))

print(f"Average MSE (Controlled):   {jnp.mean(mse_ctrl):.6f}")
print(f"Average MSE (Uncontrolled): {jnp.mean(mse_unc):.6f}")

# --- 6. Visualization ---
plt.figure(figsize=(16, 10))

# Helper to draw obstacles
def draw_obstacles(ax):
    for obs in OBSTACLES:
        # Draw physical obstacle (Red)
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.3)
        ax.add_patch(circle)
        # Draw safety margin (Dotted Red)
        margin = plt.Circle((obs[0], obs[1]), obs[2] + R_safe, color='red', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(margin)

# 1. Error Distribution
plt.subplot(2, 2, 1)
plt.boxplot([mse_ctrl, mse_unc], labels=['Controlled', 'Uncontrolled'])
plt.title(f'Tracking MSE Distribution (N={N_eval})')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Sample Index for visualization
sample_idx = 0

# 2. Agent Trajectories (Controlled)
ax_traj = plt.subplot(2, 2, 2)
draw_obstacles(ax_traj) # Draw obstacles!
for i in range(n_agents):
    # Plot path
    plt.plot(xi_ctrl_all[sample_idx, :, i, 0], xi_ctrl_all[sample_idx, :, i, 1], alpha=0.6, color='blue')
    # Plot start/end
    plt.scatter(xi_ctrl_all[sample_idx, 0, i, 0], xi_ctrl_all[sample_idx, 0, i, 1], c='green', s=10, marker='x')
    plt.scatter(xi_ctrl_all[sample_idx, -1, i, 0], xi_ctrl_all[sample_idx, -1, i, 1], c='blue', s=20)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Controlled Trajectories vs Obstacles')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# 3. Target State
plt.subplot(2, 3, 4)
plt.imshow(z_target_batch[sample_idx], origin='lower', extent=[0,1,0,1], cmap='inferno')
draw_obstacles(plt.gca())
plt.title('Target Field')
plt.colorbar()

# 4. Controlled Final State
plt.subplot(2, 3, 5)
plt.imshow(z_ctrl_all[sample_idx, -1], origin='lower', extent=[0,1,0,1], cmap='inferno')
draw_obstacles(plt.gca())
# Overlay Final Agent Positions
plt.scatter(xi_ctrl_all[sample_idx, -1, :, 0], xi_ctrl_all[sample_idx, -1, :, 1], 
            c='cyan', s=30, edgecolors='white', label='Agents')
plt.title(f'Controlled Final (MSE={mse_ctrl[sample_idx]:.4f})')
plt.colorbar()

# 5. Uncontrolled Final State
plt.subplot(2, 3, 6)
plt.imshow(z_unc_all[sample_idx, -1], origin='lower', extent=[0,1,0,1], cmap='inferno')
draw_obstacles(plt.gca())
plt.scatter(xi_unc_all[sample_idx, -1, :, 0], xi_unc_all[sample_idx, -1, :, 1], 
            c='grey', s=30, edgecolors='white', alpha=0.5)
plt.title(f'Uncontrolled Final (MSE={mse_unc[sample_idx]:.4f})')
plt.colorbar()

plt.tight_layout()
plt.savefig('heat2d_obstacles_results.png')
print("Comparison plot saved to 'heat2d_obstacles_results.png'")