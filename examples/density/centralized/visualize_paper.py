"""
NS2D Shape Formation - Publication-Quality Visualization (Adapted Layout)

Creates a figure matching the "2D Heat Equation - Decentralized" style:
- Top Row: Snapshots (Linear time spacing)
- Bottom Row: MSE, Agent Speed, and Control Intensity metrics
"""

import sys
from pathlib import Path

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import flax.serialization

from examples.density.centralized.dynamics import unroll_controlled
from examples.density.centralized.train import (
    N_AGENTS, T_STEPS, PUSH_MAX, SIGMA_PUSH, BUOYANCY, FEATURES
)
from models.policy_ns2d import NS2DControlNet


# =============================================================================
# Academic Style
# =============================================================================

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


# =============================================================================
# Helper Functions
# =============================================================================

def rollout_uncontrolled(smoke_init, xi_init, rho_target, T_steps, Nx, Ny, dt, buoyancy):
    """Rollout with zero control inputs (natural dynamics only)."""
    from examples.density.centralized.dynamics import ns2d_step_jax
    
    def step_fn(carry, _):
        smoke, xi = carry
        n = xi.shape[0]
        # Zero push velocity (no control)
        push_vel = jnp.zeros((n, 2))
        
        smoke_new = ns2d_step_jax(
            smoke, xi, push_vel,
            dt=dt, buoyancy=buoyancy,
            sigma_push=SIGMA_PUSH,
            Nx=Nx, Ny=Ny
        )
        return (smoke_new, xi), (smoke_new, xi, push_vel)
    
    _, (smoke_traj, xi_traj, v_traj) = jax.lax.scan(
        step_fn, (smoke_init, xi_init), None, length=T_steps
    )
    return smoke_traj, xi_traj, v_traj


def get_linear_timesteps(T_steps, n_points=6):
    """
    Generate linearly-spaced timesteps (Requested update).
    """
    return np.linspace(0, T_steps-1, n_points, dtype=int)


# =============================================================================
# Main Paper Visualization
# =============================================================================

def create_paper_figure(
    smoke_unctrl, smoke_ctrl, xi_traj_ctrl, vel_traj_ctrl,
    rho_target, mse_ctrl, mse_unctrl,
    filename='ns2d_paper_visualization.pdf'
):
    """
    Create publication figure.
    - Top Row: Snapshots (Linear spacing, Aspect Preserved)
    - Bottom Row: Metrics
    """
    setup_style()
    
    # Get dimensions
    T_steps, Nx, Ny = smoke_ctrl.shape
    
    # Calculate physical aspect ratio
    # Assuming dx = dy, the physical aspect is Ny/Nx.
    domain_aspect = Ny / Nx
    y_extent = 1.0 * domain_aspect 
    extent = [0, 1, 0, y_extent]
    
    # Linear timesteps
    timesteps = get_linear_timesteps(T_steps, n_points=6)
    n_cols = len(timesteps)
    
    # Metrics calculations
    vel_mags = np.sqrt(np.sum(vel_traj_ctrl**2, axis=-1))
    avg_speed_ctrl = np.mean(vel_mags, axis=1)
    max_intensity_ctrl = np.max(vel_mags, axis=1)
    
    # Color scales
    vmin = min(float(smoke_ctrl.min()), float(smoke_unctrl.min()), 0)
    vmax = max(float(smoke_ctrl.max()), float(smoke_unctrl.max()), 0.8)
    u_min, u_max = float(vel_mags.min()), float(vel_mags.max())
    
    # ─────────────────────────────────────────────────────────────────────────
    # LAYOUT CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────

    fig_width = 15
    fig_height = 8.0 

    # Vertical margins
    margin_top = 0.95
    margin_bottom = 0.08

    # Row margins
    dpc_margin_left = 0.12   
    dpc_margin_right = 0.87  
    metrics_margin_left = 0.12
    metrics_margin_right = 0.87

    # Row Heights
    dpc_row_height_ratio = 1.4     
    metrics_row_height_ratio = 0.35 
    
    # Gap
    vspace_between_rows = -0.25 

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Calculate grid positions
    avail_height = margin_top - margin_bottom
    total_units = dpc_row_height_ratio + metrics_row_height_ratio + vspace_between_rows
    unit_height = avail_height / total_units

    metrics_bottom = margin_bottom
    metrics_top = metrics_bottom + metrics_row_height_ratio * unit_height
    dpc_bottom = metrics_top + vspace_between_rows * unit_height
    dpc_top = dpc_bottom + dpc_row_height_ratio * unit_height

    # Top Grid (Snapshots)
    gs_dpc = fig.add_gridspec(
        1, n_cols,
        left=dpc_margin_left, right=dpc_margin_right,
        bottom=dpc_bottom, top=dpc_top,
        wspace=0.15,
    )

    # Bottom Grid (Metrics)
    gs_metrics = fig.add_gridspec(
        1, 3,
        left=metrics_margin_left, right=metrics_margin_right,
        bottom=metrics_bottom, top=metrics_top,
        wspace=0.25,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Top Row: Snapshots
    # ─────────────────────────────────────────────────────────────────────────
    
    row_left, row_right, row_top, row_pos = None, None, None, None
    title_fontprops = None

    for col_idx, t in enumerate(timesteps):
        ax = fig.add_subplot(gs_dpc[0, col_idx])
        
        smoke_t = smoke_ctrl[t]
        
        # aspect='equal' prevents stretching
        im = ax.imshow(smoke_t.T, origin='lower', extent=extent,
                       cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                       aspect='equal') 
        
        # Contours
        contours = ax.contour(
            np.linspace(0, 1, Nx), 
            np.linspace(0, y_extent, Ny),
            rho_target.T,
            levels=[0.3], colors='lime', linestyles='--', linewidths=1.0, alpha=0.9
        )
        contours.set_path_effects([
            patheffects.SimpleLineShadow(offset=(0.6, -0.6), shadow_color='black', alpha=0.3),
            patheffects.Normal(),
        ])

        # Agents
        xi_t = xi_traj_ctrl[t]
        vel_mags_t = vel_mags[t]
        
        ax.scatter(xi_t[:, 0], xi_t[:, 1], c=vel_mags_t, cmap='viridis', 
                   norm=Normalize(vmin=u_min, vmax=u_max),
                   s=25, edgecolors='black', linewidths=0.5, zorder=10)

        ax.set_title(f't={t}', fontsize=10)
        ax.set_xlabel('x', fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        
        # Handle Y-ticks
        if col_idx > 0:
            ax.set_yticks([])
        else:
            yticks = [0, y_extent/2, y_extent]
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y:.1f}" if y != 0 else "0" for y in yticks])
            ax.set_ylabel('y', fontsize=10)
            
        if col_idx == 0:
            row_pos = ax.get_position()
            row_left, row_top = row_pos.x0, row_pos.y1
            title_fontprops = ax.title.get_fontproperties()
        if col_idx == n_cols - 1:
            row_right = ax.get_position().x1

    # Row Title
    if row_left and row_right:
        fig.text((row_left + row_right) / 2, row_top + 0.02, 'Controlled Evolution',
                 ha='center', va='bottom', fontproperties=title_fontprops, fontweight='bold')

    # Colorbars
    if row_pos:
        cbar_height = row_pos.height
        cbar_y0 = row_pos.y0
        cbar_width = 0.012
        cbar_gap = 0.008
        
        # Left Colorbar (Control)
        cax2 = fig.add_axes([row_left - 3*cbar_gap - cbar_width, cbar_y0, cbar_width, cbar_height])
        cb2 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=u_min, vmax=u_max), cmap='viridis'), cax=cax2)
        cb2.ax.yaxis.set_ticks_position('left')
        cb2.ax.yaxis.set_label_position('right')
        cb2.set_label('Control |v|', fontsize=9)
        
        # Right Colorbar (Density)
        cax1 = fig.add_axes([row_right + cbar_gap, cbar_y0, cbar_width, cbar_height])
        cb1 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='RdBu_r'), cax=cax1)
        cb1.set_label('Smoke Density', fontsize=9)

    # ─────────────────────────────────────────────────────────────────────────
    # Bottom Row: Metrics
    # ─────────────────────────────────────────────────────────────────────────
    
    # MSE
    ax_mse = fig.add_subplot(gs_metrics[0, 0])
    ax_mse.plot(mse_unctrl, 'b-', lw=1.5, label='Uncontrolled', alpha=0.8)
    ax_mse.plot(mse_ctrl, 'r-', lw=1.5, label='Controlled', alpha=0.8)
    ax_mse.fill_between(np.arange(len(mse_ctrl)), mse_ctrl, mse_unctrl, alpha=0.15, color='green')
    ax_mse.set_xlabel('Time Step')
    ax_mse.set_ylabel('MSE')
    ax_mse.set_yscale('log')
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(fontsize=8)
    ax_mse.set_title('Tracking Error', fontsize=11, fontweight='bold')

    # Speed
    ax_speed = fig.add_subplot(gs_metrics[0, 1])
    ax_speed.plot(avg_speed_ctrl, 'g-', lw=1.5, alpha=0.8)
    ax_speed.set_xlabel('Time Step')
    ax_speed.set_ylabel('Speed')
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_ylim(bottom=0)
    ax_speed.set_title('Agent Speed', fontsize=11, fontweight='bold')

    # Intensity
    ax_ctrl = fig.add_subplot(gs_metrics[0, 2])
    ax_ctrl.plot(max_intensity_ctrl, 'm-', lw=1.5, alpha=0.8)
    ax_ctrl.set_xlabel('Time Step')
    ax_ctrl.set_ylabel('Max |v|')
    ax_ctrl.grid(True, alpha=0.3)
    ax_ctrl.set_ylim(bottom=0)
    ax_ctrl.set_title('Max Control Input', fontsize=11, fontweight='bold')

    # Save
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("NS2D Shape Formation - Paper Visualization (Final)")
    print("="*60)
    
    setup_style()
    
    # Load config
    data_dir = Path(__file__).parent.parent / 'data'
    config = np.load(data_dir / 'config.npz', allow_pickle=True)
    Nx = int(config['Nx'])
    Ny = int(config['Ny'])
    dt = float(config['dt'])
    n_agents = N_AGENTS
    
    print(f"\nGrid: {Nx}x{Ny}, Agents: {n_agents}")
    
    # Load test data
    test_data = np.load(data_dir / 'test_data.npz', allow_pickle=True)
    
    # Load model
    model = NS2DControlNet(features=FEATURES, v_max=PUSH_MAX)
    params_path = Path(__file__).parent / 'ns2d_params.msgpack'
    
    if not params_path.exists():
        print(f"Error: {params_path} not found. Run train.py first!")
        return
    
    with open(params_path, 'rb') as f:
        dummy_smoke = jnp.zeros((Nx, Ny))
        dummy_xi = jnp.zeros((n_agents, 2))
        params = model.init(jax.random.PRNGKey(0), dummy_smoke, dummy_smoke, dummy_xi)
        params = flax.serialization.from_bytes(params, f.read())
    
    print("✓ Loaded trained parameters")
    
    # Agent grid
    n_side = int(np.sqrt(n_agents))
    xi_init = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.15, 0.85, n_side),
        jnp.linspace(0.15, 1.0, n_side)
    ), axis=-1).reshape(-1, 2)
    
    T_steps = T_STEPS
    save_dir = Path(__file__).parent
    
    # Process 2 samples
    n_samples = min(2, len(test_data['rho_init']))
    
    for sample_idx in range(n_samples):
        print(f"\n{'='*60}")
        print(f"Processing Sample {sample_idx + 1}")
        print("="*60)
        
        smoke_init = jnp.array(test_data['rho_init'][sample_idx])
        rho_target = jnp.array(test_data['rho_target'][sample_idx])
        
        # Controlled trajectory
        print("▶ Running controlled simulation...")
        smoke_ctrl, xi_ctrl, vel_ctrl = unroll_controlled(
            smoke_init, xi_init, rho_target, params, model.apply, T_steps,
            Nx=Nx, Ny=Ny, dt=dt, buoyancy=BUOYANCY,
            sigma_push=SIGMA_PUSH, push_max=PUSH_MAX
        )
        
        # Uncontrolled trajectory
        print("▶ Running uncontrolled simulation...")
        smoke_unctrl, xi_unctrl, _ = rollout_uncontrolled(
            smoke_init, xi_init, rho_target, T_steps, Nx, Ny, dt, BUOYANCY
        )
        
        # Convert to numpy
        smoke_ctrl = np.array(smoke_ctrl)
        smoke_unctrl = np.array(smoke_unctrl)
        xi_ctrl = np.array(xi_ctrl)
        vel_ctrl = np.array(vel_ctrl)
        rho_target_np = np.array(rho_target)
        
        # Compute MSE over time
        mse_ctrl = np.mean((smoke_ctrl - rho_target_np)**2, axis=(1, 2))
        mse_unctrl = np.mean((smoke_unctrl - rho_target_np)**2, axis=(1, 2))
        
        # Create paper figure (Fixed Call)
        print("▶ Creating paper figure...")
        create_paper_figure(
            smoke_unctrl, smoke_ctrl, xi_ctrl, vel_ctrl,
            rho_target_np, mse_ctrl, mse_unctrl,
            filename=str(save_dir / f'ns2d_paper_sample_{sample_idx+1}.pdf')
        )
    
    # Print summary
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print("\nGenerated files:")
    for i in range(n_samples):
        print(f"  - ns2d_paper_sample_{i+1}.pdf")


if __name__ == "__main__":
    main()