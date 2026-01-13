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
from pathlib import Path

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_ks2d import DecentralizedKS2DControlNet 
from data_utils import get_batch_initial_conditions

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION (Matched to Memorization Experiment)
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'N_grid': 64,
    'L_domain': 32.0,
    'dt': 0.05,            
    
    # Must match training exactly
    'substeps': 5,         
    'n_agents': 49,         
    
    # Visualization Timeline 
    'T_chaos_steps': 50,    
    'T_control_steps': 50, 
    
    # Snapshots to display (Physical Time)
    # T=0 is control ON. 
    # Horizon is 150 * 20 * 0.005 = 15.0 seconds
    'snapshot_times': [-2.0, 0.0, 2.0, 5.0, 14.0], 
    
    'params_file': 'ks2d_centralized_params.msgpack'
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIMULATION & DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_params(model, filepath):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found.")
        
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(42)
    dummy_u = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    init_params = model.init(key, dummy_u, dummy_u, dummy_xi)
    
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def get_zero_policy():
    def zero_policy_fn(params, u, u_target, xi):
        return jnp.zeros((xi.shape[0],))
    return zero_policy_fn

def generate_transition_data(key, model, params):
    """Simulates: Chaos -> Transition -> Stabilization."""
    
    # 1. Setup Dynamics
    dyn_control = PDEDynamics2D(policy_apply_fn=model.apply)
    dyn_chaos   = PDEDynamics2D(policy_apply_fn=get_zero_policy())
    
    # 2. Get Initial Chaotic State
    print("  [Sim] Generating initial state...")
    u0 = get_batch_initial_conditions(key, 1, CONFIG['N_grid'], CONFIG['L_domain'])[0]
    
    xi_fixed = get_actuator_grid()
    u_target = jnp.zeros_like(u0)
    
    # 3. Phase 1: Run Uncontrolled
    print(f"  [Sim] Running Chaos Phase...")
    u_traj_chaos, _, _, _ = dyn_chaos.unroll_controlled(
        u0, xi_fixed, u_target, params,
        t_steps=CONFIG['T_chaos_steps'],
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'], 
        sigma=2.5 
    )
    
    # 4. Phase 2: Run Controlled
    u_handoff = u_traj_chaos[-1]
    
    print(f"  [Sim] Running Control Phase...")
    u_traj_ctrl, _, u_force_ctrl, _ = dyn_control.unroll_controlled(
        u_handoff, xi_fixed, u_target, params,
        t_steps=CONFIG['T_control_steps'],
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'], 
        sigma=2.5 # <--- CRITICAL: Match Training Sigma (was 1.0)
    )
    
    # 5. Stitch
    u_full = jnp.concatenate([u_traj_chaos, u_traj_ctrl], axis=0)
    
    dt_effective = CONFIG['substeps'] * CONFIG['dt']
    n_chaos = CONFIG['T_chaos_steps']
    n_ctrl = CONFIG['T_control_steps']
    
    t_chaos = (jnp.arange(n_chaos) - n_chaos) * dt_effective
    t_ctrl  = jnp.arange(n_ctrl) * dt_effective
    t_full = jnp.concatenate([t_chaos, t_ctrl])
    
    return t_full, u_full, u_force_ctrl

def get_actuator_grid():
    """Reconstructs the 7x7 grid."""
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_academic_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.titlesize": 16,
    })

def plot_2d_transition(t_full, u_full, u_force_ctrl, save_name="ks2d_transition.png"):
    setup_academic_style()
    
    energy = jnp.mean(u_full**2, axis=(1,2))
    
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5], hspace=0.35)
    
    # --- Row 1: Snapshots ---
    target_times = np.array(CONFIG['snapshot_times'])
    snap_indices = []
    
    for t_req in target_times:
        idx = (np.abs(t_full - t_req)).argmin()
        snap_indices.append(idx)
        
    gs_snaps = gridspec.GridSpecFromSubplotSpec(1, len(snap_indices), subplot_spec=gs[0], wspace=0.1)
    
    vmin, vmax = -3.0, 3.0
    
    for i, idx in enumerate(snap_indices):
        ax = fig.add_subplot(gs_snaps[i])
        u_snap = u_full[idx]
        t_snap = t_full[idx]
        
        im = ax.imshow(u_snap, extent=[0, CONFIG['L_domain'], 0, CONFIG['L_domain']], 
                       origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        if i == 0:
            ax.set_ylabel(r"$y$")
            ax.set_yticks([0, CONFIG['L_domain']])
        else:
            ax.set_yticks([])
            
        ax.set_xticks([0, CONFIG['L_domain']])
        ax.set_xlabel(r"$x$")
        
        if t_snap < -1e-3:
            status = "Chaos"
            color = 'firebrick'
        elif t_snap > 1e-3:
            status = "Control ON"
            color = 'navy'
        else:
            status = "Switching"
            color = 'black'
            
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
            
        ax.set_title(f"t = {t_snap:.1f}s\n{status}", color=color, fontweight='bold')
        
        xi = get_actuator_grid()
        ax.scatter(xi[:,0], xi[:,1], c='k', s=20, alpha=0.3) # Increased size slightly for 7x7

    cax = fig.add_axes([0.92, 0.55, 0.015, 0.3])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r"Vorticity $u(x,y)$")

    # --- Row 2: Energy ---
    ax_ts = fig.add_subplot(gs[1])
    
    mask_chaos = t_full <= 0
    mask_ctrl = t_full >= 0
    
    ax_ts.plot(t_full[mask_chaos], energy[mask_chaos], color='firebrick', lw=2, label='Uncontrolled')
    ax_ts.plot(t_full[mask_ctrl], energy[mask_ctrl], color='navy', lw=2, label='Controlled')
    
    ax_ts.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax_ts.set_yscale('log')
    ax_ts.set_xlim(t_full[0], t_full[-1])
    ax_ts.set_ylabel(r"Energy $\langle u^2 \rangle$")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.legend(loc='upper right')
    ax_ts.grid(True, which='both', linestyle='--', alpha=0.3)
    ax_ts.set_title("(b) System Stabilization", loc='left', fontweight='bold')

    plt.suptitle(f"2D KS Stabilization (Substeps={CONFIG['substeps']}, Grid={CONFIG['n_agents']})", y=0.98, fontsize=18)
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_name}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("--- 2D KS Visualization Script (Memorization Check) ---")
    
    model = DecentralizedKS2DControlNet(
        features=(64, 128), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=2.0
    )
    
    try:
        print(f"Loading params from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
    except Exception as e:
        print(f"Error: {e}")
        print("Run training first to generate parameters.")
        sys.exit(1)
        
    key = jax.random.PRNGKey(42)
    t_full, u_full, u_force = generate_transition_data(key, model, params)
    
    plot_2d_transition(t_full, u_full, u_force)