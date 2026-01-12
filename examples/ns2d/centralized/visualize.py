"""
Visualization for NS2D Shape Formation Control Results

Creates conference-quality figures showing:
- Controlled vs uncontrolled (natural) evolution
- Agent trajectories
- Control signals
- Tracking error comparison
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
import flax.serialization
from examples.ns2d.centralized.dynamics import unroll_controlled
from examples.ns2d.centralized.train import (
    N_AGENTS, T_STEPS, U_MAX, V_MAX, SIGMA, FEATURES
)
from models.policy_ns2d import NS2DControlNet


# =============================================================================
# Academic Style
# =============================================================================

def setup_academic_style():
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 18,
        "font.size": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 20,
        "figure.titlesize": 24,
        "axes.linewidth": 1.5,
    }
    plt.rcParams.update(tex_fonts)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_evolution_comparison(
    smoke_init, smoke_traj, rho_target, xi_traj,
    smoke_traj_unctrl=None,
    filename='ns2d_evolution.png'
):
    """Plot controlled evolution with agent positions."""
    setup_academic_style()
    
    T = smoke_traj.shape[0]
    n_plots = 6
    indices = [int(i * (T - 1) / (n_plots - 1)) for i in range(n_plots)]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.25, hspace=0.3)
    
    vmax = max(float(smoke_traj.max()), float(rho_target.max()), 0.5)
    
    Nx, Ny = smoke_traj.shape[1], smoke_traj.shape[2]
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1.25, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    levels = np.linspace(0, vmax, 50)
    n_agents = xi_traj.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    cf = None
    for i, idx in enumerate(indices):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        
        smoke_data = np.array(smoke_traj[idx])
        cf = ax.contourf(X, Y, smoke_data, levels=levels, cmap='hot', extend='max')
        ax.contour(X, Y, smoke_data, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        # Agent positions
        for j in range(n_agents):
            xi_pos = xi_traj[idx, j]
            ax.scatter(xi_pos[0], xi_pos[1], s=100, c=[colors[j]], 
                      marker='o', edgecolors='white', linewidth=2, zorder=10)
        
        # Target contour (outline)
        ax.contour(X, Y, np.array(rho_target), levels=[0.3], colors='cyan', 
                  linestyles='--', linewidths=2, alpha=0.7)
        
        ax.set_title(f'$t = {idx}$', fontweight='bold')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_aspect('equal')
    
    # Colorbar
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label(r'Smoke Density $\rho$', fontsize=18)
    
    plt.suptitle('NS2D Shape Formation: Controlled Evolution', 
                fontsize=24, fontweight='bold', y=0.98)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()


def plot_control_signals(intensity_traj, vel_traj, xi_traj, filename='ns2d_controls.png'):
    """Plot control signals and agent trajectories."""
    setup_academic_style()
    
    T = intensity_traj.shape[0]
    n_agents = intensity_traj.shape[1]
    time = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Injection intensity
    ax1 = axes[0, 0]
    for j in range(n_agents):
        ax1.plot(time, intensity_traj[:, j], color=colors[j], lw=2, label=f'Agent {j+1}')
    ax1.set_title('Injection Intensity $u_i(t)$', fontweight='bold')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Intensity')
    ax1.grid(True, alpha=0.3)
    
    # Velocity magnitude
    ax2 = axes[0, 1]
    vel_mag = np.linalg.norm(vel_traj, axis=-1)
    for j in range(n_agents):
        ax2.plot(time, vel_mag[:, j], color=colors[j], lw=2)
    ax2.set_title('Velocity Magnitude $|v_i(t)|$', fontweight='bold')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Speed')
    ax2.grid(True, alpha=0.3)
    
    # Agent trajectories (x-y plot)
    ax3 = axes[1, 0]
    for j in range(n_agents):
        ax3.plot(xi_traj[:, j, 0], xi_traj[:, j, 1], color=colors[j], lw=2)
        ax3.scatter(xi_traj[0, j, 0], xi_traj[0, j, 1], s=80, c=[colors[j]], marker='o', zorder=5)
        ax3.scatter(xi_traj[-1, j, 0], xi_traj[-1, j, 1], s=100, c=[colors[j]], 
                   marker='s', edgecolors='black', linewidth=2, zorder=5)
    ax3.set_title('Agent Trajectories $\\xi_i(t)$', fontweight='bold')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.25])
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Agent x-position over time
    ax4 = axes[1, 1]
    for j in range(n_agents):
        ax4.plot(time, xi_traj[:, j, 0], color=colors[j], lw=2, label=f'Agent {j+1}')
    ax4.set_title('Agent Position $x_i(t)$', fontweight='bold')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('$x$ position')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("NS2D Shape Formation - Visualization")
    print("="*60)
    
    # Load grid config from data
    data_dir = Path(__file__).parent.parent / 'data'
    config = np.load(data_dir / 'config.npz')
    Nx = int(config['Nx'])
    Ny = int(config['Ny'])
    dt = float(config['dt'])
    buoyancy = float(config['buoyancy'])
    
    # Use imported constants from train.py
    n_agents = N_AGENTS
    sigma = SIGMA
    
    print(f"\nGrid: {Nx}x{Ny}, Agents: {n_agents}")
    
    test_data = np.load(data_dir / 'test_data.npz')
    
    # Load model with same parameters as training
    model = NS2DControlNet(features=FEATURES, u_max=U_MAX, v_max=V_MAX)
    
    params_path = Path(__file__).parent / 'ns2d_params.msgpack'
    if not params_path.exists():
        print(f"Trained params not found: {params_path}")
        print("Run train.py first!")
        return
    
    with open(params_path, 'rb') as f:
        dummy_smoke = jnp.zeros((Nx, Ny))
        dummy_xi = jnp.zeros((n_agents, 2))
        params = model.init(jax.random.PRNGKey(0), dummy_smoke, dummy_smoke, dummy_xi)
        params = flax.serialization.from_bytes(params, f.read())
    
    print("Loaded trained parameters")
    
    # Test sample
    idx = 0
    smoke_init = jnp.array(test_data['rho_init'][idx])
    rho_target = jnp.array(test_data['rho_target'][idx])
    
    # Agent grid (matching train.py)
    n_side = int(np.sqrt(n_agents))
    xi_init = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.15, 0.85, n_side),
        jnp.linspace(0.15, 1.0, n_side)
    ), axis=-1).reshape(-1, 2)
    
    # Run controlled simulation with same T_STEPS as training
    T_steps = T_STEPS
    print(f"\nRunning controlled simulation (T={T_steps})...")
    
    smoke_traj, xi_traj, intensity_traj, vel_traj = unroll_controlled(
        smoke_init, xi_init, rho_target, params, model.apply, T_steps,
        Nx=Nx, Ny=Ny, dt=dt, buoyancy=buoyancy, sigma=sigma,
        u_max=U_MAX, v_max=V_MAX
    )
    
    smoke_traj = np.array(smoke_traj)
    xi_traj = np.array(xi_traj)
    intensity_traj = np.array(intensity_traj)
    vel_traj = np.array(vel_traj)
    
    print(f"Smoke range: [{smoke_traj.min():.3f}, {smoke_traj.max():.3f}]")
    
    # Visualize
    save_dir = Path(__file__).parent
    
    plot_evolution_comparison(
        np.array(smoke_init), smoke_traj, np.array(rho_target), xi_traj,
        filename=str(save_dir / 'ns2d_evolution.png')
    )
    
    plot_control_signals(
        intensity_traj, vel_traj, xi_traj,
        filename=str(save_dir / 'ns2d_controls.png')
    )
    
    # Initial/Final/Target comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.array(smoke_init).T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Initial', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(smoke_traj[-1].T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Final (Controlled)', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    im = axes[2].imshow(np.array(rho_target).T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Target', fontsize=16, fontweight='bold')
    axes[2].axis('off')
    
    plt.colorbar(im, ax=axes, shrink=0.8, label='Smoke Density')
    plt.suptitle('NS2D Shape Formation Result', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'ns2d_result.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: ns2d_result.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
