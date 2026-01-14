"""
Exact Result Visualization for Trained Turbulence Policy.
Compares [Uncontrolled Baseline] vs [Trained Policy] on the exact training IC.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import pickle
import flax.serialization
from pathlib import Path

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_turb import DecentralizedTurbulenceNet 

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'N_grid': 128,
    'L_domain': 1.0,
    
    # --- 1. Physics Matching Training ---
    'dt': 0.01,           
    'substeps': 10,       
    'viscosity': 1e-4,    
    
    'n_agents': 64,
    'grid_shape': (8, 8),
    'sigma': 0.04,        # Added Sigma for correct actuator physics
    
    # --- 2. Duration Settings ---
    'T_chaos_steps': 50,    # -0.50s to 0.00s
    'T_control_steps': 200, #  0.00s to 2.00s
    
    # --- 3. Snapshot Selection ---
    'snapshot_times': [-0.25, 0.0, 0.5, 1.0, 2.0],
    
    # Files
    'params_file': 'turbulence_params.msgpack',
    'ic_filename': 'turbulence_chaotic_ics_64.pkl',
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def get_actuator_grid():
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

def load_exact_training_ic():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data" 
    file_path = data_dir / CONFIG['ic_filename']
    
    if not file_path.exists():
        file_path = Path(CONFIG['ic_filename']) # Check current dir

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {CONFIG['ic_filename']}.")

    print(f"[Data] Loading ICs from: {file_path}")
    with open(file_path, 'rb') as f:
        u_pool = pickle.load(f)
    
    return jnp.array(u_pool[0]) 

def load_params(model, filepath):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found.")
        
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(42)
    xi_fixed = get_actuator_grid()
    dummy_obs = jnp.zeros((1, CONFIG['N_grid'], CONFIG['N_grid']))
    init_params = model.init(key, xi_fixed, dummy_obs)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def get_zero_policy(n_agents):
    def zero_policy_fn(params, xi, obs):
        return jnp.zeros((n_agents,))
    return zero_policy_fn

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION LOOP (Comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison(w0_hat, model, params):
    """
    Simulates:
      1. Chaos Phase (-0.5s -> 0.0s) [Both]
      2. Control Phase (0.0s -> 2.0s) [Policy]
      3. Decay Phase   (0.0s -> 2.0s) [Baseline]
    """
    n_chaos = CONFIG['T_chaos_steps']
    n_ctrl = CONFIG['T_control_steps']
    xi_fixed = get_actuator_grid()
    
    # Define Dynamics
    dyn_control = PDEDynamics2D(policy_apply_fn=model.apply)
    dyn_baseline = PDEDynamics2D(policy_apply_fn=get_zero_policy(CONFIG['n_agents']))
    
    # --- Step 1: Run Chaos Phase (Pre-Control) ---
    print(f"  [Sim] Running Chaos Phase ({n_chaos} steps)...")
    w_chaos, _ = dyn_baseline.unroll_controlled(
        w0_hat, xi_fixed, params,
        t_steps=n_chaos, # <--- Uses Chaos Steps
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        viscosity=CONFIG['viscosity'],
        actuator_grid_shape=CONFIG['grid_shape'],
        # sigma=CONFIG['sigma']
    )
    
    # Prepare Handoff (Physical -> Spectral)
    w_handoff_phys = w_chaos[-1]
    w_handoff_hat = jnp.fft.fft2(w_handoff_phys)
    
    # --- Step 2: Run Controlled Branch ---
    print(f"  [Sim] Running Controlled Branch ({n_ctrl} steps)...")
    w_ctrl_phase, u_ctrl_phase = dyn_control.unroll_controlled(
        w_handoff_hat, xi_fixed, params,
        t_steps=n_ctrl, # <--- Uses Control Steps
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        viscosity=CONFIG['viscosity'],
        actuator_grid_shape=CONFIG['grid_shape'],
        # sigma=CONFIG['sigma']
    )

    # --- Step 3: Run Baseline Branch ---
    print(f"  [Sim] Running Baseline Branch ({n_ctrl} steps)...")
    w_base_phase, _ = dyn_baseline.unroll_controlled(
        w_handoff_hat, xi_fixed, params,
        t_steps=n_ctrl,
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        viscosity=CONFIG['viscosity'],
        actuator_grid_shape=CONFIG['grid_shape'],
        # sigma=CONFIG['sigma']
    )
    
    # --- Step 4: Stitch and Time Axis ---
    w_blue = jnp.concatenate([w_chaos, w_ctrl_phase], axis=0)
    w_grey = jnp.concatenate([w_chaos, w_base_phase], axis=0)
    
    # Construct Time Axis (Start negative, cross 0)
    dt = CONFIG['dt']
    t_chaos = (np.arange(n_chaos) - n_chaos) * dt 
    t_ctrl = np.arange(n_ctrl) * dt
    t_axis = np.concatenate([t_chaos, t_ctrl])
    
    return t_axis, w_blue, w_grey, u_ctrl_phase

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_exact_result(t, w_ctrl, w_base, save_name="exact_training_result.png"):
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    
    # Calculate Enstrophy
    e_ctrl = jnp.mean(w_ctrl**2, axis=(1,2))
    e_base = jnp.mean(w_base**2, axis=(1,2))
    
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.6], hspace=0.3)
    
    # --- Row 1: Snapshots (Controlled) ---
    snap_indices = []
    for t_req in CONFIG['snapshot_times']:
        # Find closest index
        idx = (np.abs(t - t_req)).argmin()
        snap_indices.append(idx)
        
    gs_snaps = gridspec.GridSpecFromSubplotSpec(1, len(snap_indices), subplot_spec=gs[0], wspace=0.05)
    
    max_w = jnp.max(jnp.abs(w_ctrl[0])) * 0.9 # Slightly tighter bound
    
    for i, idx in enumerate(snap_indices):
        ax = fig.add_subplot(gs_snaps[i])
        w_snap = w_ctrl[idx]
        t_snap = t[idx]
        
        im = ax.imshow(w_snap, extent=[0,1,0,1], origin='lower', cmap='RdBu_r', vmin=-max_w, vmax=max_w)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Determine Phase Color
        if t_snap < -1e-4:
            status = "Chaos"
            color = 'firebrick'
        elif t_snap > 1e-4:
            status = "Control ON"
            color = 'navy'
        else:
            status = "Switching"
            color = 'black'
            
        ax.set_title(f"t = {t_snap:.2f}s", color=color, fontweight='bold')
        
        # Border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
            
        # Actuators
        xi = get_actuator_grid()
        ax.scatter(xi[:,0], xi[:,1], c='k', s=5, alpha=0.3, marker='.')

        if i == 0:
            ax.set_ylabel("Controlled\nFlow", fontsize=14, rotation=90)

    # Colorbar
    cax = fig.add_axes([0.91, 0.55, 0.01, 0.3])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Vorticity")

    # --- Row 2: Enstrophy Comparison ---
    ax_plot = fig.add_subplot(gs[1])
    
    # Mask for coloring plot
    mask_chaos = t <= 0
    mask_ctrl = t >= 0
    
    # Plot Chaos Phase (Same for both)
    ax_plot.plot(t[mask_chaos], e_base[mask_chaos], color='firebrick', lw=2)
    
    # Plot Split Branches
    ax_plot.plot(t[mask_ctrl], e_base[mask_ctrl], color='grey', linestyle='--', label='Uncontrolled (Viscous Decay)', lw=2)
    ax_plot.plot(t[mask_ctrl], e_ctrl[mask_ctrl], color='navy', label='Decentralized Policy (Ours)', lw=2)
    
    ax_plot.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    ax_plot.set_yscale('log')
    ax_plot.set_xlabel("Time (s)")
    ax_plot.set_ylabel(r"Enstrophy $\mathcal{E}(t)$")
    ax_plot.set_xlim(t[0], t[-1])
    ax_plot.legend(loc='upper right')
    ax_plot.grid(True, which="both", ls="--", alpha=0.3)
    
    ax_plot.set_title("Stabilization Performance (Exact Training Example)", loc='left', fontweight='bold')
    
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {save_name}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("--- Visualizing Exact Training Example ---")
    
    model = DecentralizedTurbulenceNet(
        features=(32, 64), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=75.0
    )
    
    try:
        params = load_params(model, CONFIG['params_file'])
        w0_hat = load_exact_training_ic()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    t_axis, w_ctrl, w_base, _ = run_comparison(w0_hat, model, params)
    
    plot_exact_result(t_axis, w_ctrl, w_base)