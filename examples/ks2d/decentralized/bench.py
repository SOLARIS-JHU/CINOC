"""
Evaluation Script: KS-2D Stabilization
Compares the trained Decentralized Policy against Uncontrolled Chaos.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
import sys
import pickle
from pathlib import Path
from functools import partial

# Ensure JAX uses 64-bit precision (Must match training)
jax.config.update("jax_enable_x64", True)

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# Imports
from dynamics_dual import PDEDynamics2D
from models.policy_ks2d import DecentralizedKS2DControlNet 
from data_utils import get_batch_initial_conditions

# --- 1. Configuration (Must Match Training) ---
CONFIG = {
    'N_grid': 64,         
    'L_domain': 32.0,      
    'dt': 0.005,
    'substeps': 20,       # Physics steps per Control step
    'T_steps': 50,        # Number of Control steps
    'n_agents': 100,      
    'N_eval': 32          # Smaller batch for 2D visualization
}

# --- 2. Helper: Zero Policy (Uncontrolled) ---
def zero_policy_apply(params, u_obs, u_target, xi_fixed):
    """
    Dummy policy for KS-2D with Fixed Actuators.
    Returns ONLY zero forcing (stabilization inactive).
    """
    # xi_fixed has shape (n_agents, 2) inside the vmap
    n_agents = xi_fixed.shape[0]
    
    # RETURN SINGLE ARRAY (Forcing only), not a tuple.
    return jnp.zeros((n_agents,)) 

# --- 3. Data Generation ---
print(f"Generating {CONFIG['N_eval']} Chaotic 2D Initial Conditions...")
key = jax.random.PRNGKey(999) # Validation Seed

# KS "Spin-up" to get realistic chaotic states
key, subkey = jax.random.split(key)
u_init_batch = get_batch_initial_conditions(
    subkey, CONFIG['N_eval'], CONFIG['N_grid'], CONFIG['L_domain']
)
u_target_batch = jnp.zeros_like(u_init_batch) # Stabilization target is 0

# Fixed Actuators (Grid)
grid_dim = int(jnp.sqrt(CONFIG['n_agents']))
x_lin = jnp.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
xv, yv = jnp.meshgrid(x_lin, x_lin)
xi_fixed_single = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
xi_fixed_batch = jnp.tile(xi_fixed_single, (CONFIG['N_eval'], 1, 1))

# --- 4. Load Model ---
model = DecentralizedKS2DControlNet(
    features=(64, 128), 
    domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
    u_max=5.0
)

print("Loading trained parameters...")
try:
    with open('ks2d_centralized_params.msgpack', 'rb') as f:
        serialized_bytes = f.read()
except FileNotFoundError:
    print("Error: 'ks2d_centralized_params.msgpack' not found.")
    sys.exit(1)

# Init dummy structure
dummy_key = jax.random.PRNGKey(0)
dummy_u = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
dummy_params = model.init(dummy_key, dummy_u, dummy_u, dummy_xi)

params = flax.serialization.from_bytes(dummy_params, serialized_bytes)

# --- 5. Evaluation Loop ---
dynamics_ctrl = PDEDynamics2D(policy_apply_fn=model.apply)
dynamics_unc = PDEDynamics2D(policy_apply_fn=zero_policy_apply)

print("Running simulations (This may take a moment due to 2D Physics)...")

@jax.jit
def run_comparison(u_init, xi_fixed, u_target):
    # Common args
    args = {
        't_steps': CONFIG['T_steps'],
        'substeps': CONFIG['substeps'],
        'N_grid': CONFIG['N_grid'],
        'L': CONFIG['L_domain'],
        'dt': CONFIG['dt'],
        'sigma': 1.2
    }
    
    # Controlled
    u_c, _, _, _ = dynamics_ctrl.unroll_controlled(
        u_init, xi_fixed, u_target, params, **args
    )
    # Uncontrolled
    u_u, _, _, _ = dynamics_unc.unroll_controlled(
        u_init, xi_fixed, u_target, params, **args
    )
    return u_c, u_u

# Batch execution
u_ctrl_all, u_unc_all = jax.vmap(run_comparison)(u_init_batch, xi_fixed_batch, u_target_batch)

# --- 6. Metrics ---
print("Calculating metrics...")

# Calculate Energy (L2 Norm squared) over time
# Shape: (N_eval, T_steps, N_grid, N_grid)
energy_ctrl = jnp.mean(u_ctrl_all**2, axis=(2, 3)) # Average over space
energy_unc = jnp.mean(u_unc_all**2, axis=(2, 3))

final_energy_ctrl = energy_ctrl[:, -1]
final_energy_unc = energy_unc[:, -1]

print(f"Mean Final Energy (Controlled):   {jnp.mean(final_energy_ctrl):.6f}")
print(f"Mean Final Energy (Uncontrolled): {jnp.mean(final_energy_unc):.6f}")

# --- 7. Plotting ---
plt.figure(figsize=(16, 10))

# 1. Energy Evolution
plt.subplot(2, 2, 1)
time_axis = jnp.arange(CONFIG['T_steps']) * CONFIG['substeps'] * CONFIG['dt']
plt.plot(time_axis, jnp.mean(energy_ctrl, axis=0), 'b-', label='Controlled', linewidth=2)
plt.plot(time_axis, jnp.mean(energy_unc, axis=0), 'r-', label='Uncontrolled', linewidth=2)
plt.yscale('log')
plt.title('System Energy Evolution (Log Scale)')
plt.xlabel('Time (s)')
plt.ylabel('Energy (L2)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Final Energy Distribution
plt.subplot(2, 2, 2)
plt.boxplot([final_energy_ctrl, final_energy_unc], labels=['Controlled', 'Uncontrolled'])
plt.yscale('log')
plt.title('Final Energy Distribution')
plt.grid(True, alpha=0.3)

# 3. Visual Snapshot (Controlled)
sample_idx = 0
plt.subplot(2, 3, 4)
plt.imshow(u_ctrl_all[sample_idx, 0], origin='lower', cmap='RdBu_r', vmin=-3, vmax=3)
plt.title('Start (t=0)')

plt.subplot(2, 3, 5)
plt.imshow(u_ctrl_all[sample_idx, CONFIG['T_steps']//2], origin='lower', cmap='RdBu_r', vmin=-3, vmax=3)
plt.title(f'Controlled (t={time_axis[len(time_axis)//2]:.1f}s)')

plt.subplot(2, 3, 6)
plt.imshow(u_ctrl_all[sample_idx, -1], origin='lower', cmap='RdBu_r', vmin=-3, vmax=3)
plt.title(f'Controlled (t={time_axis[-1]:.1f}s)')

plt.tight_layout()
plt.savefig('ks2d_comparison_results.png')
print("Comparison plot saved to 'ks2d_comparison_results.png'")