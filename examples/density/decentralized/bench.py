"""
Evaluation Script: Decentralized NS2D Shape Formation
Compares the trained Neural Policy against a Zero-Control baseline.
Metric: Holding Loss (l_hold) as defined in training.
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
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# CHANGED: Import unroll_with_full_loss instead of unroll_controlled
from examples.density.decentralized.dynamics import unroll_with_full_loss
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
# ADDED: R_SAFE is required for the full loss function signature
R_SAFE = 0.15       

# --- 2. Helper: Zero Policy (Baseline) ---
def zero_policy_apply(params, rho, rho_target, xi):
    """
    Dummy policy: Fans are off (Zero Velocity).
    Returns: (u_force_x, u_force_y) for each agent.
    """
    n_batch = xi.shape[0]
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

# Initialize Agents
n_side = int(np.sqrt(N_AGENTS))
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

dummy_key = jax.random.PRNGKey(0)
dummy_rho = jnp.zeros((Nx, Ny))
dummy_xi = jnp.zeros((N_AGENTS, 2))
init_params = model.init(dummy_key, dummy_rho, dummy_rho, dummy_xi)
params = flax.serialization.from_bytes(init_params, serialized_bytes)

# --- 5. Evaluation Loop ---
print("Running simulations (Comparing l_hold)...")

@jax.jit
def run_comparison(rho_init, xi_init, rho_target):
    # Sim args must match unroll_with_full_loss signature exactly
    sim_kwargs = {
        'Nx': Nx, 'Ny': Ny, 
        'n_agents': N_AGENTS,  # explicitly needed for full loss
        'dt': dt, 
        'buoyancy': BUOYANCY, 
        'sigma_push': SIGMA_PUSH, 
        'push_max': PUSH_MAX,
        'R_safe': R_SAFE
    }
    
    # Controlled Run
    # Returns: smoke_final, xi_final, l_hold, l_coll, l_bound, l_smooth, l_effort, l_mass
    ret_c = unroll_with_full_loss(
        rho_init, xi_init, rho_target, params, model.apply, T_STEPS, **sim_kwargs
    )
    # We capture Final State + l_hold (3rd return value)
    rho_c_final, xi_c_final, loss_c = ret_c[0], ret_c[1], ret_c[2]
    
    # Uncontrolled Run
    ret_u = unroll_with_full_loss(
        rho_init, xi_init, rho_target, params, zero_policy_apply, T_STEPS, **sim_kwargs
    )
    rho_u_final, xi_u_final, loss_u = ret_u[0], ret_u[1], ret_u[2]
    
    return (rho_c_final, xi_c_final, loss_c), (rho_u_final, xi_u_final, loss_u)

# Batch Execution
(out_ctrl, out_unc) = jax.vmap(run_comparison)(rho_init_batch, xi_init_batch, rho_target_batch)

# Unpack results
# Shapes: rho=(N, Nx, Ny), xi=(N, Agents, 2), loss=(N,)
rho_ctrl_final, xi_ctrl_final, l_hold_ctrl = out_ctrl
rho_unc_final,  xi_unc_final,  l_hold_unc  = out_unc

# --- 6. Metrics (Holding Loss) ---
print("Calculating metrics...")

print(f"Mean Holding Loss (Controlled):   {jnp.mean(l_hold_ctrl):.6f}")
print(f"Mean Holding Loss (Uncontrolled): {jnp.mean(l_hold_unc):.6f}")

# --- 7. Plotting ---
plt.figure(figsize=(16, 10))

# 1. Error Distribution
plt.subplot(2, 2, 1)
plt.boxplot([l_hold_ctrl, l_hold_unc], labels=['Controlled', 'Uncontrolled'])
plt.title(f'Holding Loss Comparison (N={N_EVAL})')
plt.ylabel('l_hold (Lower is better)')
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
plt.imshow(rho_ctrl_final[sample_idx].T, origin='lower', cmap='Blues', extent=[0,1,0,1])
# Overlay Agents
cx = xi_ctrl_final[sample_idx, :, 0]
cy = xi_ctrl_final[sample_idx, :, 1]
plt.scatter(cx, cy, c='orange', s=40, edgecolors='black', label='Fans')
plt.title(f'Controlled Final (Loss={l_hold_ctrl[sample_idx]:.4f})')
plt.axis('off')

# Uncontrolled Final
plt.subplot(2, 3, 6)
plt.imshow(rho_unc_final[sample_idx].T, origin='lower', cmap='Blues', extent=[0,1,0,1])
ux = xi_unc_final[sample_idx, :, 0]
uy = xi_unc_final[sample_idx, :, 1]
plt.scatter(ux, uy, c='gray', s=40, edgecolors='black', alpha=0.5)
plt.title(f'Uncontrolled Final (Loss={l_hold_unc[sample_idx]:.4f})')
plt.axis('off')

# 3. Agent Displacements (Controlled)
# Note: unroll_with_full_loss only returns Start and End positions, not full trajectory.
# We plot arrows to show movement.
plt.subplot(2, 2, 2)
plt.imshow(rho_target_batch[sample_idx].T, origin='lower', cmap='Blues', alpha=0.3, extent=[0,1,0,1])

start_pts = xi_init_batch[sample_idx]
end_pts = xi_ctrl_final[sample_idx]

plt.scatter(start_pts[:, 0], start_pts[:, 1], marker='x', c='k', s=30, label='Start')
plt.scatter(end_pts[:, 0], end_pts[:, 1], marker='o', c='orange', edgecolors='k', s=30, label='End')

# Draw arrows
for i in range(N_AGENTS):
    plt.arrow(
        start_pts[i, 0], start_pts[i, 1],
        end_pts[i, 0] - start_pts[i, 0], end_pts[i, 1] - start_pts[i, 1],
        color='orange', alpha=0.6, length_includes_head=True, head_width=0.02
    )

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Fan Displacement (Controlled)')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
save_path = Path(__file__).parent / 'ns2d_comparison_l_hold.png'
plt.savefig(save_path)
print(f"Comparison plot saved to {save_path}")