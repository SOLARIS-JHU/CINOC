"""
2D Heat Equation Control with Obstacles - Decentralized Visualization
Creates publication-quality figures comparing controlled vs uncontrolled evolution
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
import sys
from pathlib import Path
import flax.serialization
import numpy as np

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

script_dir = Path(__file__).resolve().parent.parent.parent.parent
script_path = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
import data_utils

# Obstacle configuration: [x_center, y_center, radius]
OBSTACLES = jnp.array([
    [0.15, 0.50, 0.08],   # Left middle (outside agent grid)
    [0.85, 0.50, 0.08],   # Right middle (outside agent grid)
    [0.50, 0.15, 0.08],   # Bottom middle (outside agent grid)
])

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

def draw_obstacles(ax, obstacles):
    """Draw circular obstacles on axis."""
    for obs in obstacles:
        x, y, r = obs
        circle = Circle((x, y), r, color='gray', alpha=0.5,
                       edgecolor='black', linewidth=1.5, zorder=5)
        ax.add_patch(circle)

def load_params(model, filepath, n_grid=32, n_agents=16):
    """Load trained parameters from msgpack file."""
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((n_grid, n_grid))
    dummy_xi = jnp.zeros((n_agents, 2))
    init_params = model.init(key, dummy_z, dummy_z, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def rollout_uncontrolled(z_init, xi_init, T_steps):
    """Rollout with zero control inputs."""
    from tesseracts.solverHeat2D_centralized import solver

    def step_fn(carry, _):
        z_curr, xi_curr = carry
        u_zero = jnp.zeros(xi_curr.shape[0])
        v_zero = jnp.zeros_like(xi_curr)
        z_next, xi_next = solver.adi_step(z_curr, xi_curr, u_zero, v_zero)
        return (z_next, xi_next), (z_next, xi_next, u_zero, v_zero)

    _, (z_traj, xi_traj, u_traj, v_traj) = jax.lax.scan(
        step_fn, (z_init, xi_init), None, length=T_steps
    )
    return z_traj, xi_traj, u_traj, v_traj

def get_log_timesteps(T_steps, n_points=6):
    """
    Generate logarithmically-spaced timesteps over the full time horizon.
    """
    timesteps = np.logspace(0, np.log10(T_steps-1), n_points, dtype=int)
    timesteps = np.unique(timesteps)  # Remove duplicates
    timesteps[0] = 0  # Ensure we start at 0
    timesteps[-1] = T_steps-1  # Ensure last is final step
    return timesteps

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  2D HEAT EQUATION WITH OBSTACLES - DECENTRALIZED VISUALIZATION")
    print("=" * 70)

    setup_style()

    # Configuration
    n_grid = 32
    n_agents = 16
    T_steps = 300

    # Initialize model and dynamics (no Tesseract)
    model = DecentralizedHeat2DControlNet(features=(16, 32))
    dynamics = PDEDynamics(None, policy_apply_fn=model.apply, use_tesseract=False)

    params_path = script_path / 'decentralized_params_heat2d_obstacles.msgpack'
    output_dir = script_path / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        params = load_params(model, params_path, n_grid, n_agents)
        print(f"✓ Loaded trained parameters ({n_agents} agents)")
    except FileNotFoundError:
        print(f"✗ Error: {params_path} not found")
        return

    # Generate single test scenario (using scenario 1 from original)
    print("\n▶ Generating test scenario...")
    key = jax.random.PRNGKey(1234)
    key, k1, k2 = jax.random.split(key, 3)

    xx, yy, z_init = data_utils.generate_grf_2d(k1, n_points=n_grid)
    _, _, z_target = data_utils.generate_grf_2d(k2, n_points=n_grid)

    # Initialize agents in grid pattern
    n_side = int(jnp.sqrt(n_agents))
    spacing = 0.8 / (n_side + 1)
    xi_init = []
    for i in range(n_side):
        for j in range(n_side):
            if len(xi_init) < n_agents:
                xi_init.append([0.1 + spacing*(i+1), 0.1 + spacing*(j+1)])
    xi_init = jnp.array(xi_init)

    print(f"Obstacles at: {OBSTACLES.tolist()}")
    print("▶ Running controlled trajectory...")
    z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl, v_traj_ctrl = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )

    print("▶ Running uncontrolled trajectory...")
    z_traj_unctrl, xi_traj_unctrl, u_traj_unctrl, v_traj_unctrl = rollout_uncontrolled(
        z_init, xi_init, T_steps
    )

    print("✓ Trajectories generated")

    # Compute metrics
    mse_ctrl = jnp.mean((z_traj_ctrl - z_target[None, :, :])**2, axis=(1, 2))
    mse_unctrl = jnp.mean((z_traj_unctrl - z_target[None, :, :])**2, axis=(1, 2))

    # Agent speeds (magnitude of velocity)
    speeds_ctrl = jnp.sqrt(jnp.sum(v_traj_ctrl**2, axis=-1))  # (T, n_agents)
    avg_speed_ctrl = jnp.mean(speeds_ctrl, axis=1)  # (T,)

    # Control intensity (mean absolute control)
    control_intensity = jnp.mean(jnp.abs(u_traj_ctrl), axis=1)  # (T,)

    # Get log-spaced timesteps for field plots
    timesteps = get_log_timesteps(T_steps, n_points=6)
    n_cols = len(timesteps)

    print(f"\n▶ Creating visualization at timesteps: {timesteps}")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYOUT CONFIGURATION - Independent control of row positioning
    # ─────────────────────────────────────────────────────────────────────────

    # Figure dimensions
    fig_width = 15
    fig_height = 6.1

    # Overall vertical margins (shared)
    margin_top = 0.94
    margin_bottom = 0.06

    # Per-row horizontal margins (independent control)
    dpc_margin_left = 0.15
    dpc_margin_right = 0.84
    metrics_margin_left = 0.13
    metrics_margin_right = 0.88

    # DPC Row (Top) - Controls positioning of field evolution plots
    dpc_row_height_ratio = 1.0        # Relative height of DPC row
    dpc_row_hspace = 0.17             # Horizontal spacing within DPC row
    dpc_row_vpos_ratio = 0.55         # Vertical position ratio (0.55 = 55% from bottom)

    # Metrics Row (Bottom) - Controls positioning of 3 metric subplots
    metrics_row_height_ratio = 0.26   # Relative height of metrics row
    metrics_row_hspace = 0.24         # Horizontal spacing between the 3 metric plots
    metrics_row_vpos_ratio = 0.05     # Vertical position ratio from bottom

    # Vertical gap between rows
    vspace_between_rows = -0.23

    # ─────────────────────────────────────────────────────────────────────────

    # Create figure: DPC evolution row + metrics row (separate gridspecs per row)
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Compute vertical allocation (normalized figure coords)
    avail_height = margin_top - margin_bottom
    total_units = dpc_row_height_ratio + metrics_row_height_ratio + vspace_between_rows
    unit_height = avail_height / total_units

    metrics_bottom = margin_bottom
    metrics_top = metrics_bottom + metrics_row_height_ratio * unit_height
    dpc_bottom = metrics_top + vspace_between_rows * unit_height
    dpc_top = dpc_bottom + dpc_row_height_ratio * unit_height

    # Gridspec for DPC row (top)
    gs_dpc = fig.add_gridspec(
        1,
        n_cols,
        left=dpc_margin_left,
        right=dpc_margin_right,
        bottom=dpc_bottom,
        top=dpc_top,
        wspace=dpc_row_hspace,
    )

    # Gridspec for metrics row (bottom)
    gs_metrics = fig.add_gridspec(
        1,
        3,
        left=metrics_margin_left,
        right=metrics_margin_right,
        bottom=metrics_bottom,
        top=metrics_top,
        wspace=metrics_row_hspace,
    )

    # Determine global color scale
    vmin = min(jnp.min(z_init), jnp.min(z_target),
               jnp.min(z_traj_ctrl), jnp.min(z_traj_unctrl))
    vmax = max(jnp.max(z_init), jnp.max(z_target),
               jnp.max(z_traj_ctrl), jnp.max(z_traj_unctrl))

    # Control intensity color scale
    u_min = jnp.min(u_traj_ctrl)
    u_max = jnp.max(u_traj_ctrl)

    z_target_np = np.array(z_target)
    target_min = float(jnp.min(z_target))
    target_max = float(jnp.max(z_target))
    contour_levels = np.linspace(target_min, target_max, 7)

    # Plot field snapshots
    row_left = None
    row_right = None
    row_top = None
    row_pos = None
    title_fontprops = None
    for col_idx, t in enumerate(timesteps):
        # Row 1: DPC Controlled Evolution
        ax = fig.add_subplot(gs_dpc[0, col_idx])
        im = ax.imshow(z_traj_ctrl[t], origin='lower', extent=[0, 1, 0, 1],
                       cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
        contours = ax.contour(
            z_target_np,
            levels=contour_levels,
            origin='lower',
            extent=[0, 1, 0, 1],
            cmap='RdBu_r',
            linestyles='--',
            linewidths=1.0,
            alpha=0.9,
        )
        contours.set_path_effects([
            patheffects.SimpleLineShadow(offset=(0.6, -0.6), shadow_color='black', alpha=0.3),
            patheffects.Normal(),
        ])
        draw_obstacles(ax, OBSTACLES)

        # Overlay actuators with control intensity as color
        u_colors = u_traj_ctrl[t]
        norm = Normalize(vmin=u_min, vmax=u_max)
        scatter = ax.scatter(xi_traj_ctrl[t, :, 0], xi_traj_ctrl[t, :, 1],
                             c=u_colors, cmap='viridis', norm=norm,
                             s=25, edgecolors='black', linewidths=0.5, zorder=10)

        ax.set_title(f't={t}', fontsize=10, fontweight='normal')
        ax.set_xlabel('x', fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if col_idx > 0:
            ax.set_yticklabels([])
        if col_idx == 0:
            row_pos = ax.get_position()
            row_left = row_pos.x0
            row_top = row_pos.y1
            title_fontprops = ax.title.get_fontproperties()
        if col_idx == n_cols - 1:
            row_right = ax.get_position().x1

    if row_left is not None and row_right is not None and row_top is not None:
        title_y = min(row_top + 0.03, 0.97)
        fig.text(
            (row_left + row_right) / 2,
            title_y,
            'DPC Controlled Evolution',
            ha='center',
            va='bottom',
            fontproperties=title_fontprops,
            fontweight='bold',
        )

    # Add colorbars (aligned to DPC row, matched to square height)
    cbar_gap = 0.008
    cbar_width = 0.014
    if row_pos is not None and row_left is not None and row_right is not None:
        cbar_height = row_pos.height
        cbar_y0 = row_pos.y0
        cax2 = fig.add_axes([row_left - 4*cbar_gap - cbar_width, cbar_y0, cbar_width, cbar_height])
        cax1 = fig.add_axes([row_right + cbar_gap, cbar_y0, cbar_width, cbar_height])
    else:
        cax2 = fig.add_axes([0.1, 0.7, cbar_width, 0.2])
        cax1 = fig.add_axes([0.9, 0.7, cbar_width, 0.2])
    cb1 = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='RdBu_r'),
        cax=cax1,
        label='Temperature',
    )
    cb1.ax.tick_params(labelsize=8)

    # Control intensity colorbar
    cb2 = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=u_min, vmax=u_max), cmap='viridis'),
        cax=cax2,
        label='Control u',
    )
    cb2.ax.yaxis.set_ticks_position('left')
    cb2.ax.yaxis.set_label_position('right')
    cb2.ax.tick_params(labelsize=8, labelleft=True, labelright=False, pad=1)
    cb2.ax.yaxis.set_label_coords(1.15, 0.5)
    cb2.set_label('Control u', labelpad=4)

    # Row 2: Time-series metrics (3 subplots) - uses metrics_row_hspace for independent control
    # MSE Tracking Error
    ax_mse = fig.add_subplot(gs_metrics[0, 0])
    ax_mse.plot(mse_unctrl, 'b-', lw=1.5, label='Uncontrolled', alpha=0.8)
    ax_mse.plot(mse_ctrl, 'r-', lw=1.5, label='DPC Controlled', alpha=0.8)
    time_err = np.arange(len(mse_ctrl))
    ax_mse.fill_between(time_err, mse_ctrl, mse_unctrl, alpha=0.15, color='green')
    ax_mse.set_xlabel('Time Step', fontsize=10)
    ax_mse.set_ylabel('MSE', fontsize=10)
    ax_mse.set_title('MSE Tracking Error', fontsize=11, fontweight='bold')
    ax_mse.set_yscale('log')
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(fontsize=9, loc='center right')

    # Agent Speed
    ax_speed = fig.add_subplot(gs_metrics[0, 1])
    ax_speed.plot(avg_speed_ctrl, 'g-', lw=1.5, alpha=0.8)
    ax_speed.set_xlabel('Time Step', fontsize=10)
    ax_speed.set_ylabel('Avg Speed', fontsize=10)
    ax_speed.set_title('Agent Speed', fontsize=11, fontweight='bold')
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_ylim(bottom=0)

    # Control Intensity
    ax_control = fig.add_subplot(gs_metrics[0, 2])
    ax_control.plot(control_intensity, 'm-', lw=1.5, alpha=0.8)
    ax_control.set_xlabel('Time Step', fontsize=10)
    ax_control.set_ylabel('Avg |u|', fontsize=10)
    ax_control.set_title('Control Intensity', fontsize=11, fontweight='bold')
    ax_control.grid(True, alpha=0.3)
    ax_control.set_ylim(bottom=0)

    # Save as PDF (vector graphics)
    pdf_path = output_dir / 'heat2d_obstacles_decentralized_visualization.pdf'
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {pdf_path}")

    # Also save as high-res PNG
    png_path = output_dir / 'heat2d_obstacles_decentralized_visualization.png'
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Print final metrics
    print(f"\n{'─'*70}")
    print(f"  FINAL METRICS")
    print(f"{'─'*70}")
    print(f"  Final MSE (Controlled):   {mse_ctrl[-1]:.6f}")
    print(f"  Final MSE (Uncontrolled): {mse_unctrl[-1]:.6f}")
    print(f"  Improvement:              {(1 - mse_ctrl[-1]/mse_unctrl[-1])*100:.1f}%")
    print(f"{'─'*70}")

    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
