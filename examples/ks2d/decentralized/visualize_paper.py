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
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'N_grid': 64,
    'L_domain': 32.0,
    
    # --- Physics ---
    'dt': 0.005,
    'substeps': 20,          
    'n_agents': 100,         
    'sigma': 1.2,            
    
    # --- Duration ---
    'T_chaos_steps': 20,     # 2.0s of pure chaos
    'T_control_steps': 100,  # 10.0s of comparison
    
    # Snapshot times relative to control start (t=0)
    'snapshot_times': [-1.0, 0.0, 2.0, 5.0, 9.0],
    
    # --- Files ---
    'params_file': 'ks2d_centralized_params.msgpack',
}

def get_actuator_grid():
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def get_zero_policy():
    def zero_policy_fn(params, u, u_target, xi):
        return jnp.zeros((xi.shape[0],))
    return zero_policy_fn

def run_comparison(u0, model, params):
    """Run Chaos -> Branch(Control vs Baseline)."""
    n_chaos = CONFIG['T_chaos_steps']
    n_ctrl = CONFIG['T_control_steps']
    xi_fixed = get_actuator_grid()
    u_target = jnp.zeros_like(u0)
    
    dyn_control = PDEDynamics2D(policy_apply_fn=model.apply)
    dyn_baseline = PDEDynamics2D(policy_apply_fn=get_zero_policy())
    
    # 1. Chaos Phase
    print(f"  [Sim] 1. Running Chaos Phase...")
    u_chaos, _, _, _ = dyn_baseline.unroll_controlled(
        u0, xi_fixed, u_target, params,
        t_steps=n_chaos, substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        sigma=CONFIG['sigma']
    )
    
    u_handoff = u_chaos[-1]
    
    # 2. Controlled Branch
    print(f"  [Sim] 2. Running Control Branch...")
    u_ctrl_phase, _, _, _ = dyn_control.unroll_controlled(
        u_handoff, xi_fixed, u_target, params,
        t_steps=n_ctrl, substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        sigma=CONFIG['sigma']
    )

    # 3. Baseline Branch
    print(f"  [Sim] 3. Running Natural Branch...")
    u_base_phase, _, _, _ = dyn_baseline.unroll_controlled(
        u_handoff, xi_fixed, u_target, params,
        t_steps=n_ctrl, substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        sigma=CONFIG['sigma']
    )
    
    # Stitch
    u_controlled_full = jnp.concatenate([u_chaos, u_ctrl_phase], axis=0)
    u_natural_full = jnp.concatenate([u_chaos, u_base_phase], axis=0)
    
    dt_eff = CONFIG['dt'] * CONFIG['substeps']
    t_chaos = (np.arange(n_chaos) - n_chaos) * dt_eff
    t_ctrl = np.arange(n_ctrl) * dt_eff
    t_axis = np.concatenate([t_chaos, t_ctrl])
    
    return t_axis, u_controlled_full, u_natural_full

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_snapshots_row(ax_list, t, u_data, row_title, v_limit):
    """Plots a single row of snapshots."""
    
    snap_indices = []
    for t_req in CONFIG['snapshot_times']:
        idx = (np.abs(t - t_req)).argmin()
        snap_indices.append(idx)
        
    im = None
    for i, idx in enumerate(snap_indices):
        ax = ax_list[i]
        u_snap = u_data[idx]
        t_snap = t[idx]
        
        im = ax.imshow(u_snap, extent=[0, CONFIG['L_domain'], 0, CONFIG['L_domain']], 
                       origin='lower', cmap='RdBu_r', vmin=-v_limit, vmax=v_limit)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Border Colors
        if t_snap < -1e-4: color = 'firebrick' # Chaos phase
        elif t_snap > 1e-4: color = 'navy'     # Branch phase
        else: color = 'black'
            
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
            
        if i == 2: 
             ax.set_title(row_title, fontsize=12, pad=10, fontweight='bold')

        ax.set_xlabel(f"t={t_snap:.1f}s", fontsize=9)

    return im

def plot_energy_comparison(ax, t, u_ctrl, u_base):
    """Plots the Energy (L2) comparison line graph."""
    e_ctrl = jnp.mean(u_ctrl**2, axis=(1,2))
    e_base = jnp.mean(u_base**2, axis=(1,2))
    
    mask_chaos = t <= 0
    mask_branch = t >= 0
    
    # 1. Chaos Phase
    ax.plot(t[mask_chaos], e_base[mask_chaos], color='firebrick', lw=2, label='Chaotic Precursor')
    
    # 2. Divergence
    ax.plot(t[mask_branch], e_base[mask_branch], color='grey', linestyle='--', lw=2, label='Natural Evolution')
    ax.plot(t[mask_branch], e_ctrl[mask_branch], color='navy', lw=2, label='Controlled')
    
    ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    
    ax.set_yscale('log')
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(r"Energy $\langle u^2 \rangle$", fontsize=11)
    ax.set_xlim(t[0], t[-1])
    
    ax.grid(True, which="major", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    
    # --- LEGEND ADJUSTMENT ---
    # Moved to 'center right' to occupy the empty space between the falling control curve 
    # and the high natural curve.
    ax.legend(fontsize=9, loc='center right', framealpha=0.9)
    ax.set_title("Stabilization Performance", fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"--- KS-2D Single Sample Comparison (L={CONFIG['L_domain']}) ---")
    
    # 1. Init Model
    model = DecentralizedKS2DControlNet(
        features=(64, 128), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=5.0
    )
    
    # 2. Load Params
    if not Path(CONFIG['params_file']).exists():
        print(f"Error: {CONFIG['params_file']} not found.")
        sys.exit(1)
        
    with open(CONFIG['params_file'], 'rb') as f:
        raw_params = f.read()
    
    dummy_u = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    init_params = model.init(jax.random.PRNGKey(0), dummy_u, dummy_u, dummy_xi)
    params = flax.serialization.from_bytes(init_params, raw_params)

    # 3. Generate ONE Random IC
    seed = 2024
    print(f"Generating IC with seed {seed}...")
    u0 = get_batch_initial_conditions(jax.random.PRNGKey(seed), 1, CONFIG['N_grid'], CONFIG['L_domain'])[0]
    
    # 4. Run Comparison
    t_axis, u_ctrl, u_base = run_comparison(u0, model, params)
    
    # 5. Setup Plot Layout
    N_snaps = len(CONFIG['snapshot_times'])
    
    fig = plt.figure(figsize=(17, 6)) # Slightly wider figure to accommodate gap
    
    # --- SPACING ADJUSTMENT ---
    # wspace=0.35 (increased from 0.15) creates the gap you requested
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.8, 1], wspace=0.35, hspace=0.05)
    
    # -- Row 1: Controlled Visuals --
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, N_snaps, subplot_spec=gs[0, 0], wspace=0.05)
    ax_row1 = [fig.add_subplot(gs_row1[j]) for j in range(N_snaps)]
    
    # -- Row 2: Natural Visuals --
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, N_snaps, subplot_spec=gs[1, 0], wspace=0.05)
    ax_row2 = [fig.add_subplot(gs_row2[j]) for j in range(N_snaps)]
    
    # -- Right: Comparison Plot --
    ax_plot = fig.add_subplot(gs[:, 1])
    
    # 6. Plotting
    v_lim = 3.0
    
    # Plot Visuals
    im = plot_snapshots_row(ax_row1, t_axis, u_ctrl, "Controlled Evolution", v_lim)
    plot_snapshots_row(ax_row2, t_axis, u_base, "Natural Evolution", v_lim)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_row1 + ax_row2, fraction=0.02, pad=0.02)
    # --- TERMINOLOGY ADJUSTMENT ---
    cbar.set_label(r'State Field $u(x,y)$', fontsize=10) 
    
    # Plot Graph
    plot_energy_comparison(ax_plot, t_axis, u_ctrl, u_base)

    # ---------------------------------------------------------
    # --- ALIGNMENT LOGIC ---
    # ---------------------------------------------------------
    fig.canvas.draw() 

    pos_top = ax_row1[-1].get_position()
    pos_bot = ax_row2[-1].get_position()
    pos_plot = ax_plot.get_position()

    new_bottom = pos_bot.y0
    new_top = pos_top.y1
    new_height = new_top - new_bottom
    
    ax_plot.set_position([pos_plot.x0, new_bottom, pos_plot.width, new_height])
    # ---------------------------------------------------------

    save_name = "ks2d_single_sample_comparison.pdf"
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {save_name}")