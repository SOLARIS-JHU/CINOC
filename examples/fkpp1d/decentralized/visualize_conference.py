"""
Conference-Quality Visualization for FKPP1D Decentralized DPC
Style reference: Times New Roman fonts, RdBu_r colormap, clean academic layout
Layout: Single Row [Uncontrolled | Controlled | Tracking Error]
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tesseract_core import Tesseract
import sys
import flax.serialization
from pathlib import Path
from functools import partial

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# ACADEMIC STYLING (Times New Roman / Serif)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_academic_style():
    """Configure matplotlib for academic/conference style."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 30,
        "font.size": 28,
        "legend.fontsize": 22,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "axes.titlesize": 32,
        "figure.titlesize": 36,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.5,
        "grid.alpha": 0.3,
        "grid.linewidth": 1.0,
    }
    plt.rcParams.update(tex_fonts)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_params(model, filepath, n_pde=100, n_agents=8):
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    init_params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def rollout_uncontrolled(z_init, xi_init, dynamics, T_steps):
    """Rollout with zero control inputs."""
    import tesseracts.solverFKPP_decentralized.solver as solver
    
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        u_zero = jnp.zeros_like(xi_curr)
        v_zero = jnp.zeros_like(xi_curr)
        z_next, xi_next = solver.fkpp_step_1d(z_curr, xi_curr, u_zero, v_zero)
        return (z_next, xi_next), z_next
    
    _, z_traj = jax.lax.scan(step_fn, (z_init, xi_init), None, length=T_steps)
    return z_traj

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_comparison_figure(x_grid, z_init, z_target, z_traj_ctrl, z_traj_unctrl, 
                             u_traj, v_traj, xi_traj, T_steps, example_idx=1):
    """
    Create a 3-panel comparison figure in academic style.
    Layout: Row 1 [Uncontrolled | Controlled | Tracking Error]
    """
    setup_academic_style()
    
    # Convert to numpy
    x = np.array(x_grid).squeeze()
    z_target_np = np.array(z_target).squeeze()
    z_ctrl = np.array(z_traj_ctrl)
    z_unctrl = np.array(z_traj_unctrl)
    xi_np = np.array(xi_traj)
    
    T = z_ctrl.shape[0]
    step = max(1, T // 12)
    plot_indices = list(range(0, T, step))
    if (T - 1) not in plot_indices:
        plot_indices.append(T - 1)
    
    cmap = plt.get_cmap("RdBu_r")
    n_agents = xi_np.shape[1]
    colors_agents = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    # CHANGED: 1 Row, 3 Columns. Adjusted figsize height to be shorter.
    fig, axes = plt.subplots(1, 3, figsize=(36, 11))
    
    label_fs = 34
    title_fs = 36
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 1: Uncontrolled Evolution (Row 1, Col 1)
    # ────────────────────────────────────────────────────────────────────────
    ax1 = axes[0]
    for t in plot_indices:
        color = cmap(t / (T - 1))
        lw = 3.5 if t in [0, T - 1] else 2.0
        alpha = 1.0 if t in [0, T - 1] else 0.6
        ax1.plot(x, z_unctrl[t], color=color, lw=lw, alpha=alpha)
    
    ax1.plot(x, z_target_np, 'k--', lw=3.0, label="Target", zorder=10)
    ax1.set_title(r"Evolution (Control = 0)", fontsize=title_fs, fontweight='bold')
    ax1.set_xlabel(r"Position $x$", fontsize=label_fs)
    ax1.set_ylabel(r"Population $z(x,t)$", fontsize=label_fs)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.1])  # FKPP bounded [0, 1]
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 2: Controlled Evolution (Row 1, Col 2)
    # ────────────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    for t in plot_indices:
        color = cmap(t / (T - 1))
        lw = 3.5 if t in [0, T - 1] else 2.0
        alpha = 1.0 if t in [0, T - 1] else 0.6
        ax2.plot(x, z_ctrl[t], color=color, lw=lw, alpha=alpha)
    
    ax2.plot(x, z_target_np, 'k--', lw=3.0, label="Target", zorder=10)
    
    # Mark final agent positions
    for j in range(n_agents):
        xi_final = float(xi_np[-1, j])
        idx = int(np.clip(xi_final * len(x), 0, len(x)-1))
        ax2.scatter(xi_final, z_ctrl[-1, idx], s=250, color=colors_agents[j], 
                   edgecolors='black', linewidth=2, zorder=15, marker='o')
    
    ax2.set_title(r"Controlled Evolution", fontsize=title_fs, fontweight='bold')
    ax2.set_xlabel(r"Position $x$", fontsize=label_fs)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.1])  # FKPP bounded [0, 1]
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 3: Tracking Error Comparison (Row 1, Col 3)
    # ────────────────────────────────────────────────────────────────────────
    ax3 = axes[2]
    mse_ctrl = np.mean((z_ctrl - z_target_np[None, :])**2, axis=1)
    mse_unctrl = np.mean((z_unctrl - z_target_np[None, :])**2, axis=1)
    time_err = np.arange(len(mse_ctrl))
    
    ax3.semilogy(time_err, mse_unctrl, 'b-', lw=3.5, label='Uncontrolled', alpha=0.8)
    ax3.semilogy(time_err, mse_ctrl, 'r-', lw=3.5, label='Controlled', alpha=0.8)
    ax3.fill_between(time_err, mse_ctrl, mse_unctrl, alpha=0.15, color='green')
    
    ax3.set_title(r"Tracking Error (MSE)", fontsize=title_fs, fontweight='bold')
    ax3.set_xlabel(r"Time step", fontsize=label_fs)
    ax3.set_ylabel(r"MSE (log scale)", fontsize=label_fs)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=26)
    
    # ────────────────────────────────────────────────────────────────────────
    # Common Legend (Simplified)
    # ────────────────────────────────────────────────────────────────────────
    combined_handles = [
        plt.Line2D([], [], color=cmap(0.0), lw=3.5, label="Initial state"),
        plt.Line2D([], [], color=cmap(1.0), lw=3.5, label="Final state"),
        plt.Line2D([], [], color='black', ls='--', lw=3.0, label="Target"),
        plt.Line2D([], [], color='white', marker='o', markeredgecolor='black', 
                  markerfacecolor='gray', markersize=15, label="Agents (Final)")
    ]
    
    fig.legend(
        handles=combined_handles,
        loc='lower center',
        ncol=4,
        fontsize=32,
        frameon=True,
        fancybox=True,
        shadow=False,
        handlelength=2.5,
        bbox_to_anchor=(0.5, -0.05)
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    save_path = f'fkpp_dpc_decentralized_row_ex{example_idx}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return save_path

def main():
    print("=" * 60)
    print("  FKPP EQUATION DECENTRALIZED DPC - CONFERENCE VISUALIZATION")
    print("=" * 60)
    
    n_pde, n_agents, T_steps = 100, 16, 300
    n_examples = 3 
    
    solver_ts = Tesseract.from_image("solver_fkpp1d_decentralized:latest")
    
    with solver_ts:
        model = DecentralizedControlNet(features=(64, 64))
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)
        
        try:
            params = load_params(model, 'decentralized_params.msgpack', n_pde, n_agents)
            print(f"✓ Loaded trained parameters ({n_agents} agents)")
        except FileNotFoundError:
            print("✗ Error: decentralized_params.msgpack not found")
            return
        
        x_grid = jnp.linspace(0, 1, n_pde)
        key = jax.random.PRNGKey(42)
        
        saved_files = []
        
        for i in range(n_examples):
            print(f"\n▶ Generating Example {i+1}/{n_examples}...")
            
            key, k1, k2 = jax.random.split(key, 3)
            _, z_init = data_utils.generate_grf(k1, n_points=n_pde, length_scale=0.15 + i*0.05)
            _, z_target = data_utils.generate_grf(k2, n_points=n_pde, length_scale=0.35 + i*0.05)
            xi_init = jnp.linspace(0.15, 0.85, n_agents)
            
            # Controlled rollout
            z_traj_ctrl, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
                z_init, xi_init, z_target, params, T_steps
            )
            
            # Uncontrolled rollout
            z_traj_unctrl = rollout_uncontrolled(z_init, xi_init, dynamics, T_steps)
            
            # Create the 1x3 comparison figure
            f1 = create_comparison_figure(
                x_grid, z_init, z_target, z_traj_ctrl, z_traj_unctrl,
                u_traj, v_traj, xi_traj, T_steps, example_idx=i+1
            )
            saved_files.append(f1)
        
        print("\n" + "=" * 60)
        print("  VISUALIZATION COMPLETE")
        print("=" * 60)
        print("\nGenerated files:")
        for f in saved_files:
            print(f"  • {f}")

if __name__ == "__main__":
    main()