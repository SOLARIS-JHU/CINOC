"""
Conference-Quality Visualization for KS-2D Centralized Control.
Generates a "Chaos -> Control" transition figure with 2D snapshots and energy metrics.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
import pickle
from pathlib import Path
from functools import partial

# Force CPU for visualization (prevents OOM on local machines usually)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- Path Setup (Matches your training script) ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_ks2d import KS2DControlNet
from data_utils import get_batch_initial_conditions

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'N_grid': 64,
    'L_domain': 32.0,
    'dt': 0.005,
    'n_agents': 16,     # 4x4 grid
    
    # Visualization Timeline
    'T_chaos': 200,     # Steps of chaos to record before t=0
    'T_control': 300,   # Steps of control after t=0
    
    # Snapshots to display (relative time)
    'snapshot_times': [-0.75, 0.0, 0.2, 0.5, 1.2], 
    
    'params_file': 'ks2d_centralized_params.msgpack'
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIMULATION & DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_params(model, filepath):
    """Loads trained Flax parameters."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found.")
        
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    # Re-init dummy to get structure
    key = jax.random.PRNGKey(0)
    dummy_u = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    init_params = model.init(key, dummy_u, dummy_u, dummy_xi)
    
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def get_zero_policy():
    """Returns a dummy policy function that outputs zero control."""
    def zero_policy_fn(params, u, u_target, xi):
        # Output shape: (n_agents,)
        return jnp.zeros((xi.shape[0],))
    return zero_policy_fn

def generate_transition_data(key, model, params):
    """
    Simulates: Chaos (Uncontrolled) -> Transition (t=0) -> Stabilization (Controlled)
    """
    # 1. Setup Dynamics
    # We use two dynamic runners: one with the trained policy, one with zero policy
    dyn_control = PDEDynamics2D(policy_apply_fn=model.apply)
    dyn_chaos   = PDEDynamics2D(policy_apply_fn=get_zero_policy())
    
    # 2. Get Initial Chaotic State
    # We generate a fresh chaotic state on the attractor
    print("  [Sim] Generating initial chaotic state...")
    u0 = get_batch_initial_conditions(key, 1, CONFIG['N_grid'], CONFIG['L_domain'])[0]
    
    # 3. Phase 1: Run Uncontrolled (Chaos)
    # We treat u0 as the start of our "record window".
    # (Note: u0 is already chaotic, so we just evolve it to visualize the chaos)
    xi_fixed = get_actuator_grid()
    u_target = jnp.zeros_like(u0)
    dummy_params = params # Not used by zero_policy, but required by API

    print(f"  [Sim] Running {CONFIG['T_chaos']} steps of uncontrolled dynamics...")
    u_traj_chaos, _, _, _ = dyn_chaos.unroll_controlled(
        u0, xi_fixed, u_target, dummy_params,
        t_steps=CONFIG['T_chaos'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'], sigma=1.0
    )
    
    # 4. Phase 2: Run Controlled
    # Start from the last state of the chaotic trajectory
    u_handoff = u_traj_chaos[-1]
    
    print(f"  [Sim] Running {CONFIG['T_control']} steps of controlled dynamics...")
    u_traj_ctrl, _, u_force_ctrl, _ = dyn_control.unroll_controlled(
        u_handoff, xi_fixed, u_target, params,
        t_steps=CONFIG['T_control'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'], sigma=1.0
    )
    
    # 5. Stitch
    # Concatenate: Chaos[:-1] + Control (to avoid duplicating the handoff frame)
    u_full = jnp.concatenate([u_traj_chaos, u_traj_ctrl], axis=0)
    
    # Create time axis
    t_chaos = jnp.arange(-CONFIG['T_chaos'], 0) * CONFIG['dt']
    t_ctrl  = jnp.arange(0, CONFIG['T_control']) * CONFIG['dt']
    t_full  = jnp.concatenate([t_chaos, t_ctrl])
    
    return t_full, u_full, u_force_ctrl

def get_actuator_grid():
    """Reconstructs the 4x4 actuator grid."""
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLOTTING (Academic Style)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_academic_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 16,
        "font.size": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 18,
    })

def plot_2d_transition(t_full, u_full, u_force_ctrl, save_name="ks2d_transition.png"):
    setup_academic_style()
    
    # --- Metrics Calculation ---
    # L2 Norm (Energy)
    energy = jnp.mean(u_full**2, axis=(1,2))
    
    # --- Figure Layout ---
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.6], hspace=0.3)
    
    # Row 1: Snapshots (Sub-grid)
    # Select indices for snapshots
    target_times = np.array(CONFIG['snapshot_times'])
    snap_indices = []
    actual_times = []
    
    for t_req in target_times:
        idx = (np.abs(t_full - t_req)).argmin()
        snap_indices.append(idx)
        actual_times.append(t_full[idx])
        
    gs_snaps = gridspec.GridSpecFromSubplotSpec(1, len(snap_indices), subplot_spec=gs[0], wspace=0.1)
    
    # Global limits for consistent colorbar
    vmin, vmax = -3.0, 3.0
    
    for i, idx in enumerate(snap_indices):
        ax = fig.add_subplot(gs_snaps[i])
        u_snap = u_full[idx]
        t_snap = t_full[idx]
        
        # Plot Field
        im = ax.imshow(u_snap, extent=[0, CONFIG['L_domain'], 0, CONFIG['L_domain']], 
                       origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        # Styling
        if i == 0:
            ax.set_ylabel(r"$y$")
            ax.set_yticks([0, CONFIG['L_domain']])
        else:
            ax.set_yticks([])
            
        ax.set_xticks([0, CONFIG['L_domain']])
        ax.set_xlabel(r"$x$")
        
        # Title/Annotation
        status = "Chaos" if t_snap < 0 else "Control"
        if abs(t_snap) < 1e-3: status = "On"
        
        # Colored border for emphasis
        color = 'red' if t_snap < 0 else 'blue'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            
        ax.set_title(f"t = {t_snap:.1f}s\n({status})", color=color, fontweight='bold')
        
        # Overlay Actuators (white dots)
        xi = get_actuator_grid()
        ax.scatter(xi[:,0], xi[:,1], c='k', s=20, alpha=0.3, label='Actuators')

    # Colorbar for Snapshots
    cax = fig.add_axes([0.92, 0.55, 0.015, 0.3]) # Right side
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r"Vorticity field $u(x,y)$")

    # Row 2: Time Series (Energy)
    ax_ts = fig.add_subplot(gs[1])
    
    # Plot Chaos Phase
    mask_chaos = t_full <= 0
    ax_ts.plot(t_full[mask_chaos], energy[mask_chaos], color='red', lw=2, label='Uncontrolled')
    
    # Plot Control Phase
    mask_ctrl = t_full >= 0
    ax_ts.plot(t_full[mask_ctrl], energy[mask_ctrl], color='blue', lw=2, label='Controlled')
    
    # Vertical line at t=0
    ax_ts.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax_ts.text(0.02, max(energy)*0.9, "Control ON", fontweight='bold')
    
    # Formatting
    ax_ts.set_yscale('log')
    ax_ts.set_xlim(t_full[0], t_full[-1])
    ax_ts.set_ylabel(r"Mean Energy $\langle u^2 \rangle$")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.grid(True, which='both', linestyle='--', alpha=0.3)
    ax_ts.legend(loc='upper right')
    ax_ts.set_title("(b) System Energy Decay", loc='left', fontweight='bold')

    plt.suptitle(f"Stabilizing 2D Kuramoto-Sivashinsky Turbulence (Centralized Control)", y=0.98, fontsize=20)
    
    plt.savefig(save_name, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved visualization to {save_name}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("--- 2D KS Visualization Script ---")
    
    # 1. Load Model
    model = KS2DControlNet(
        features=(32, 64), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=2.0
    )
    
    try:
        print(f"Loading params from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have run the training script first.")
        sys.exit(1)
        
    # 2. Generate Data
    key = jax.random.PRNGKey(42)
    t_full, u_full, u_force = generate_transition_data(key, model, params)
    
    # 3. Plot
    plot_2d_transition(t_full, u_full, u_force)