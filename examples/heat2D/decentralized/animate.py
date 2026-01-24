"""
2D Heat Equation Control - Decentralized Animation
Creates animated visualization with controlled evolution + MSE
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import sys
from pathlib import Path
import flax.serialization
import numpy as np

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_style():
    """Configure matplotlib for animation style."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 13,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titlesize": 14,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
    }
    plt.rcParams.update(tex_fonts)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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
    from tesseracts.solverHeat2D_decentralized import solver

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

# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_control_animation(z_traj_unctrl, z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl,
                             z_target, fps=30, duration=10):
    """
    Create 1×2 animation:
    [DPC Controlled Evolution]  [MSE Plot]
    """
    setup_style()

    # Convert to numpy
    z_unctrl = np.array(z_traj_unctrl)
    z_ctrl = np.array(z_traj_ctrl)
    z_target_np = np.array(z_target)
    xi_ctrl = np.array(xi_traj_ctrl)
    u_ctrl = np.array(u_traj_ctrl)

    T = z_ctrl.shape[0]

    # Compute metrics
    mse_ctrl = np.mean((z_ctrl - z_target_np[None, :, :])**2, axis=(1, 2))
    mse_unctrl = np.mean((z_unctrl - z_target_np[None, :, :])**2, axis=(1, 2))
    # Color scales
    vmin = min(z_ctrl.min(), z_target_np.min())
    vmax = max(z_ctrl.max(), z_target_np.max())
    u_min, u_max = u_ctrl.min(), u_ctrl.max()
    contour_levels = np.linspace(float(z_target_np.min()), float(z_target_np.max()), 7)

    # Calculate frames
    total_frames = fps * duration
    frame_indices = np.linspace(0, T-1, total_frames).astype(int)

    # Create figure
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[0.06, 1, 1, 0.06],
        wspace=0.4,
        left=0.08,
        right=0.95,
        top=0.93,
        bottom=0.08,
    )

    ax1 = fig.add_subplot(gs[0, 1])  # Controlled
    ax2 = fig.add_subplot(gs[0, 2])  # MSE

    # Panel 1: DPC Controlled Evolution
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel(r'Position $x$', fontsize=12)
    ax1.set_ylabel(r'Position $y$', fontsize=12)
    ax1.set_title('DPC Controlled Evolution', fontsize=13, fontweight='bold')
    im_ctrl = ax1.imshow(z_ctrl[0], origin='lower', extent=[0, 1, 0, 1],
                         cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
    contours = ax1.contour(
        z_target_np,
        levels=contour_levels,
        origin='lower',
        extent=[0, 1, 0, 1],
        cmap='RdBu_r',
        linestyles='--',
        linewidths=1.2,
        alpha=0.95,
    )
    contours.set_path_effects([
        patheffects.SimpleLineShadow(offset=(0.6, -0.6), shadow_color='black', alpha=0.3),
        patheffects.Normal(),
    ])

    # Actuators with control intensity
    norm_u = Normalize(vmin=u_min, vmax=u_max)
    scatter_ctrl = ax1.scatter([], [], c=[], cmap='viridis', norm=norm_u,
                          s=30, edgecolors='black', linewidths=0.6, zorder=10)

    # Panel 2: MSE Evolution
    ax2.set_xlim([0, T])
    ax2.set_ylim([min(mse_ctrl.min(), mse_unctrl.min())*0.5,
                  max(mse_ctrl.max(), mse_unctrl.max())*1.2])
    ax2.set_xlabel(r'Time Step', fontsize=12)
    ax2.set_ylabel(r'MSE', fontsize=12)
    ax2.set_title('MSE Tracking Error', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    line_unctrl, = ax2.plot([], [], 'b-', lw=2.5, label='Uncontrolled', alpha=0.8)
    line_ctrl, = ax2.plot([], [], 'r-', lw=2.5, label='DPC Controlled', alpha=0.8)
    time_marker = ax2.axvline(x=0, color='green', lw=2, alpha=0.7, linestyle='--')
    ax2.legend(fontsize=11, loc='center right')

    # Add colorbars (aligned to DPC panel)
    cax2 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 3])

    cb1 = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='RdBu_r'),
        cax=cax1,
        label='Temperature',
    )
    cb1.ax.tick_params(labelsize=10)

    # Control intensity colorbar (panel 2)
    cb2 = fig.colorbar(
        ScalarMappable(norm=norm_u, cmap='viridis'),
        cax=cax2,
        label='Control u',
    )
    cb2.ax.yaxis.set_ticks_position('left')
    cb2.ax.yaxis.set_label_position('right')
    cb2.ax.tick_params(labelsize=10, labelleft=True, labelright=False)

    # Title and time text
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=13, fontweight='bold')

    def init():
        im_ctrl.set_data(z_ctrl[0])
        scatter_ctrl.set_offsets(np.empty((0, 2)))
        line_unctrl.set_data([], [])
        line_ctrl.set_data([], [])
        time_marker.set_xdata([0])
        time_text.set_text('')
        return [im_ctrl, scatter_ctrl, line_unctrl, line_ctrl, time_marker, time_text]

    def animate(frame):
        t = frame_indices[frame]

        # Update field plots
        im_ctrl.set_data(z_ctrl[t])

        # Update actuators
        positions = xi_ctrl[t]
        scatter_ctrl.set_offsets(positions)
        scatter_ctrl.set_array(u_ctrl[t])

        # Update MSE lines
        line_unctrl.set_data(np.arange(t+1), mse_unctrl[:t+1])
        line_ctrl.set_data(np.arange(t+1), mse_ctrl[:t+1])
        time_marker.set_xdata([t])

        # Update time text
        time_text.set_text(f't = {t} / {T-1}')

        return [im_ctrl, scatter_ctrl, line_unctrl, line_ctrl, time_marker, time_text]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=total_frames, interval=1000/fps, blit=True)

    return fig, anim

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  2D HEAT EQUATION DPC - CENTRALIZED ANIMATION")
    print("=" * 70)

    # Configuration
    n_grid = 32
    n_agents = 16
    T_steps = 300
    fps = 30
    duration = 10  # seconds

    # Load model (native JAX inference)
    model = DecentralizedHeat2DControlNet(features=(16, 32))
    dynamics = PDEDynamics(None, policy_apply_fn=model.apply, use_tesseract=False)

    script_path = Path(__file__).resolve().parent
    params_path = script_path / 'decentralized_params_heat2d.msgpack'
    output_dir = script_path
    try:
        params = load_params(model, params_path, n_grid, n_agents)
        print(f"✓ Loaded trained parameters ({n_agents} agents)")
    except FileNotFoundError:
        print(f"✗ Error: {params_path} not found")
        return

    # Generate test scenario (same as visualize.py - scenario 1)
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

    print("▶ Running controlled trajectory...")
    z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl, v_traj_ctrl = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )

    print("▶ Running uncontrolled trajectory...")
    z_traj_unctrl, xi_traj_unctrl, u_traj_unctrl, v_traj_unctrl = rollout_uncontrolled(
        z_init, xi_init, T_steps
    )

    print(f"✓ Generated {T_steps} timesteps")

    # Create animation
    print(f"\n▶ Creating animation ({duration}s @ {fps}fps)...")
    fig, anim = create_control_animation(
        z_traj_unctrl, z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl,
        z_target, fps=fps, duration=duration
    )

    # Save as GIF
    print("▶ Saving GIF (this may take a few minutes)...")
    gif_path = output_dir / 'heat2d_animation.gif'
    anim.save(gif_path, writer='pillow', fps=fps, dpi=150)
    print(f"✓ Saved: {gif_path}")

    # Save as MP4 (higher quality)
    try:
        print("▶ Saving MP4 (high resolution)...")
        mp4_path = output_dir / 'heat2d_animation.mp4'
        anim.save(mp4_path, writer='ffmpeg', fps=fps, dpi=200,
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"✓ Saved: {mp4_path}")
    except Exception as e:
        print(f"⚠ MP4 save failed (ffmpeg may not be installed): {e}")

    plt.close()

    print("\n" + "=" * 70)
    print("  ANIMATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
