"""
NS2D Shape Formation - Conference Quality Visualization
Adapts NS2D results to the specific layout:
Row 1: Controlled Evolution Snapshots with Agent overlays
Row 2: Metrics (MSE, Speed, Intensity)
"""
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import flax.serialization

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.density.centralized.dynamics import unroll_controlled
from examples.density.centralized.train import (
    N_AGENTS, T_STEPS, PUSH_MAX, SIGMA_INJECT, SIGMA_PUSH, BUOYANCY, FEATURES
)
from models.policy_ns2d import NS2DControlNet

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_style():
    """Configure matplotlib for publication-quality figures."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.titlesize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    }
    plt.rcParams.update(tex_fonts)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_log_timesteps(T_steps, n_points=6):
    """Generate logarithmically-spaced timesteps over the full time horizon."""
    timesteps = np.logspace(0, np.log10(T_steps-1), n_points, dtype=int)
    timesteps = np.unique(timesteps)
    timesteps[0] = 0
    timesteps[-1] = T_steps-1
    return timesteps

def zero_policy(params, smoke, target, xi):
    """Dummy policy for uncontrolled rollout."""
    n = xi.shape[0]
    return jnp.zeros(n), jnp.zeros((n, 2))

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  NS2D SHAPE FORMATION - PUBLICATION VISUALIZATION")
    print("=" * 70)

    setup_style()

    # 1. Load Configuration and Data
    data_dir = Path(__file__).parent.parent / 'data'
    config = np.load(data_dir / 'config.npz')
    Nx = int(config['Nx'])
    Ny = int(config['Ny'])
    dt = float(config['dt'])
    
    test_data = np.load(data_dir / 'test_data.npz')
    
    # Load Model
    model = NS2DControlNet(features=FEATURES, v_max=PUSH_MAX)
    params_path = Path(__file__).parent / 'ns2d_params.msgpack'
    
    if not params_path.exists():
        print(f"✗ Error: {params_path} not found")
        return

    with open(params_path, 'rb') as f:
        dummy_smoke = jnp.zeros((Nx, Ny))
        dummy_xi = jnp.zeros((N_AGENTS, 2))
        params = model.init(jax.random.PRNGKey(0), dummy_smoke, dummy_smoke, dummy_xi)
        params = flax.serialization.from_bytes(params, f.read())
    
    print("✓ Loaded parameters and config")

    # 2. Setup Scenario (Using Sample 0)
    print("\n▶ Generating test scenario (Sample 0)...")
    idx = 0
    smoke_init = jnp.array(test_data['rho_init'][idx])
    rho_target = jnp.array(test_data['rho_target'][idx])
    
    # Initial grid of agents
    n_side = int(np.sqrt(N_AGENTS))
    xi_init = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.15, 0.85, n_side),
        jnp.linspace(0.15, 1.0, n_side)
    ), axis=-1).reshape(-1, 2)

    # 3. Run Controlled Trajectory
    print("▶ Running controlled trajectory...")
    smoke_traj, xi_traj, u_traj, v_traj = unroll_controlled(
        smoke_init, xi_init, rho_target, params, model.apply, T_STEPS,
        Nx=Nx, Ny=Ny, dt=dt, buoyancy=BUOYANCY,
        sigma_inject=SIGMA_INJECT, sigma_push=SIGMA_PUSH,
        u_max=0.0, push_max=PUSH_MAX
    )
    
    # Convert to numpy for plotting
    smoke_traj = np.array(smoke_traj)
    xi_traj = np.array(xi_traj)
    u_traj = np.array(u_traj)
    v_traj = np.array(v_traj)
    rho_target_np = np.array(rho_target)

    # 4. Run Uncontrolled Trajectory (for comparison metrics)
    print("▶ Running uncontrolled trajectory...")
    smoke_traj_unctrl, _, _, _ = unroll_controlled(
        smoke_init, xi_init, rho_target, None, zero_policy, T_STEPS,
        Nx=Nx, Ny=Ny, dt=dt, buoyancy=BUOYANCY,
        sigma_inject=SIGMA_INJECT, sigma_push=SIGMA_PUSH,
        u_max=0.0, push_max=0.0
    )
    smoke_traj_unctrl = np.array(smoke_traj_unctrl)

    # 5. Compute Metrics
    # MSE
    mse_ctrl = np.mean((smoke_traj - rho_target_np[None, ...])**2, axis=(1, 2))
    mse_unctrl = np.mean((smoke_traj_unctrl - rho_target_np[None, ...])**2, axis=(1, 2))
    
    # Speed (magnitude of velocity)
    speeds_ctrl = np.sqrt(np.sum(v_traj**2, axis=-1))
    avg_speed_ctrl = np.mean(speeds_ctrl, axis=1)
    
    # Control Intensity (magnitude of u)
    control_intensity = np.mean(np.abs(u_traj), axis=1)

    # 6. Visualization Setup
    timesteps = get_log_timesteps(T_STEPS, n_points=6)
    n_cols = len(timesteps)
    
    # ─────────────────────────────────────────────────────────────────────────
    # LAYOUT CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    fig_width = 15
    fig_height = 6.1

    # Margins and Positioning
    margin_top = 0.94
    margin_bottom = 0.06
    dpc_margin_left = 0.15
    dpc_margin_right = 0.84
    metrics_margin_left = 0.13
    metrics_margin_right = 0.88

    # Row Heights
    dpc_row_height_ratio = 1.0
    metrics_row_height_ratio = 0.26
    vspace_between_rows = -0.23

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Compute vertical allocation
    avail_height = margin_top - margin_bottom
    total_units = dpc_row_height_ratio + metrics_row_height_ratio + vspace_between_rows
    unit_height = avail_height / total_units

    metrics_bottom = margin_bottom
    metrics_top = metrics_bottom + metrics_row_height_ratio * unit_height
    dpc_bottom = metrics_top + vspace_between_rows * unit_height
    dpc_top = dpc_bottom + dpc_row_height_ratio * unit_height

    # Gridspecs
    gs_dpc = fig.add_gridspec(1, n_cols, left=dpc_margin_left, right=dpc_margin_right,
                              bottom=dpc_bottom, top=dpc_top, wspace=0.17)
    
    gs_metrics = fig.add_gridspec(1, 3, left=metrics_margin_left, right=metrics_margin_right,
                                  bottom=metrics_bottom, top=metrics_top, wspace=0.24)

    # Color Scales
    vmin, vmax = 0, 1.0  # Smoke density is typically normalized
    u_min, u_max = np.min(u_traj), np.max(u_traj)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROW 1: Evolution Snapshots
    # ─────────────────────────────────────────────────────────────────────────
    row_pos = None
    row_left, row_right, row_top_val = None, None, None
    title_fontprops = None

    for col_idx, t in enumerate(timesteps):
        ax = fig.add_subplot(gs_dpc[0, col_idx])
        
        # Plot Smoke Density
        # Note: extent matches NS2D domain [0, 1] x [0, 1.25]
        im = ax.imshow(smoke_traj[t].T, origin='lower', extent=[0, 1, 0, 1.25],
                       cmap='hot', vmin=vmin, vmax=vmax, interpolation='bilinear')
        
        # Overlay Target Contour
        contours = ax.contour(
            rho_target_np.T, levels=[0.3], origin='lower', extent=[0, 1, 0, 1.25],
            colors='cyan', linestyles='--', linewidths=1.0, alpha=0.9
        )
        contours.set_path_effects([
            patheffects.SimpleLineShadow(offset=(0.6, -0.6), shadow_color='black', alpha=0.3),
            patheffects.Normal(),
        ])

        # Overlay Agents
        u_colors = u_traj[t]
        norm = Normalize(vmin=u_min, vmax=u_max)
        ax.scatter(xi_traj[t, :, 0], xi_traj[t, :, 1],
                   c=u_colors, cmap='viridis', norm=norm,
                   s=25, edgecolors='white', linewidths=0.5, zorder=10)

        # Styling
        ax.set_title(f't={t}', fontsize=10, fontweight='normal')
        ax.set_xlabel('x', fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.6, 1.25])
        
        if col_idx > 0:
            ax.set_yticklabels([])
        
        # Store positions for titles/colorbars
        if col_idx == 0:
            row_pos = ax.get_position()
            row_left = row_pos.x0
            row_top_val = row_pos.y1
            title_fontprops = ax.title.get_fontproperties()
        if col_idx == n_cols - 1:
            row_right = ax.get_position().x1

    # Row Title
    if row_left and row_right:
        fig.text((row_left + row_right) / 2, min(row_top_val + 0.03, 0.97),
                 'Controlled Evolution (NS2D)', ha='center', va='bottom',
                 fontproperties=title_fontprops, fontweight='bold')

    # Colorbars
    cbar_gap = 0.008
    cbar_width = 0.014
    if row_pos:
        cbar_height = row_pos.height
        cbar_y0 = row_pos.y0
        cax2 = fig.add_axes([row_left - 4*cbar_gap - cbar_width, cbar_y0, cbar_width, cbar_height])
        cax1 = fig.add_axes([row_right + cbar_gap, cbar_y0, cbar_width, cbar_height])
        
        # Smoke Density Colorbar (Right)
        cb1 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='hot'),
                           cax=cax1, label='Smoke Density')
        cb1.ax.tick_params(labelsize=8)
        
        # Control Intensity Colorbar (Left)
        cb2 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=u_min, vmax=u_max), cmap='viridis'),
                           cax=cax2)
        cb2.ax.yaxis.set_ticks_position('left')
        cb2.ax.yaxis.set_label_position('right')
        cb2.ax.tick_params(labelsize=8, labelleft=True, labelright=False, pad=1)
        cb2.set_label('Injection u', labelpad=4)
        cb2.ax.yaxis.set_label_coords(1.15, 0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 2: Metrics
    # ─────────────────────────────────────────────────────────────────────────
    
    # 1. MSE Tracking Error
    ax_mse = fig.add_subplot(gs_metrics[0, 0])
    time_err = np.arange(len(mse_ctrl))
    ax_mse.plot(mse_unctrl, 'b-', lw=1.5, label='Uncontrolled', alpha=0.8)
    ax_mse.plot(mse_ctrl, 'r-', lw=1.5, label='Controlled', alpha=0.8)
    ax_mse.fill_between(time_err, mse_ctrl, mse_unctrl, alpha=0.15, color='green')
    
    ax_mse.set_xlabel('Time Step', fontsize=10)
    ax_mse.set_ylabel('MSE', fontsize=10)
    ax_mse.set_title('Tracking Error', fontsize=11, fontweight='bold')
    ax_mse.set_yscale('log')
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(fontsize=9, loc='upper right')

    # 2. Agent Speed
    ax_speed = fig.add_subplot(gs_metrics[0, 1])
    ax_speed.plot(avg_speed_ctrl, 'g-', lw=1.5, alpha=0.8)
    ax_speed.set_xlabel('Time Step', fontsize=10)
    ax_speed.set_ylabel('Avg Speed |v|', fontsize=10)
    ax_speed.set_title('Agent Speed', fontsize=11, fontweight='bold')
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_ylim(bottom=0)

    # 3. Control Intensity
    ax_control = fig.add_subplot(gs_metrics[0, 2])
    ax_control.plot(control_intensity, 'm-', lw=1.5, alpha=0.8)
    ax_control.set_xlabel('Time Step', fontsize=10)
    ax_control.set_ylabel('Avg |u|', fontsize=10)
    ax_control.set_title('Control Intensity', fontsize=11, fontweight='bold')
    ax_control.grid(True, alpha=0.3)
    ax_control.set_ylim(bottom=0)

    # ─────────────────────────────────────────────────────────────────────────
    # SAVING
    # ─────────────────────────────────────────────────────────────────────────
    save_dir = Path(__file__).parent
    
    png_path = save_dir / 'ns2d_publication_vis.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {png_path}")
    
    pdf_path = save_dir / 'ns2d_publication_vis.pdf'
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    
    plt.close()

    # Final Stats
    print(f"\n{'─'*70}")
    print(f"  FINAL METRICS")
    print(f"{'─'*70}")
    print(f"  Final MSE (Controlled):   {mse_ctrl[-1]:.6f}")
    print(f"  Final MSE (Uncontrolled): {mse_unctrl[-1]:.6f}")
    print(f"  Improvement:              {(1 - mse_ctrl[-1]/mse_unctrl[-1])*100:.1f}%")
    print(f"{'─'*70}")

if __name__ == "__main__":
    main()