"""
Phase 4: Agent Count Transfer Study - Visualization

Evaluates the trained model (N=100) on various agent counts.
Generates:
1. Scalability curve (MSE vs agent count)
2. Efficiency metric (MSE per agent)
3. Trajectory snapshots at different densities
4. Energy decay comparison
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.serialization
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from train_utils_ks2d import get_agent_grid, get_or_create_data, evaluate_policy
from models.policy_ks2d import DecentralizedKS2DControlNet

# Test Parameters
TEST_AGENT_COUNTS = [36, 49, 64, 81, 100, 121, 144, 169, 196]  # 6²...14² (train=100)
N_TEST_SAMPLES = 20
TRAIN_N_AGENTS = 100

# Physics parameters (must match training)
PHYSICS_CONFIG = {
    'N_grid': 64,
    'L_domain': 32.0,
    'dt': 0.005,
    'substeps': 20,
    'T_steps': 50,
    'sigma': 1.2,
}

# Clean evaluation (no noise)
NOISE_CONFIG = {
    'noise_u': 0.0,
    'noise_z': 0.0,
}


def load_model(save_dir):
    """Load trained parameters."""
    params_path = save_dir / "baseline_n100_params.msgpack"
    if not params_path.exists():
        raise FileNotFoundError(f"Model not found: {params_path}")

    with open(params_path, 'rb') as f:
        params = flax.serialization.from_bytes(None, f.read())

    return params


def evaluate_full_trajectory(params, model, u_init, n_agents, key):
    """
    Evaluates and returns full trajectory for visualization.
    """
    xi_fixed = get_agent_grid(n_agents, PHYSICS_CONFIG['L_domain'])
    u_target = jnp.zeros((PHYSICS_CONFIG['N_grid'], PHYSICS_CONFIG['N_grid']))

    from dynamics_dual import PDEDynamics2D
    dynamics = PDEDynamics2D(policy_apply_fn=model.apply)

    u_traj, xi_traj, u_ctrl_traj, _ = dynamics.unroll_controlled(
        u_init,
        xi_fixed,
        u_target,
        params,
        t_steps=PHYSICS_CONFIG['T_steps'],
        substeps=PHYSICS_CONFIG['substeps'],
        N_grid=PHYSICS_CONFIG['N_grid'],
        L=PHYSICS_CONFIG['L_domain'],
        dt=PHYSICS_CONFIG['dt'],
        sigma=PHYSICS_CONFIG['sigma'],
        key=key,
        **NOISE_CONFIG
    )

    return u_traj, xi_fixed


def plot_scalability_curve(results_df, save_dir):
    """
    Plots MSE vs agent count with shaded region at training count.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by agent count
    grouped = results_df.groupby('N_Agents')['Final_MSE'].agg(['mean', 'std', 'min', 'max'])

    ax.errorbar(
        grouped.index,
        grouped['mean'],
        yerr=grouped['std'],
        marker='o',
        markersize=8,
        capsize=5,
        color='blue',
        label='Mean ± Std'
    )

    # Fill between min/max
    ax.fill_between(
        grouped.index,
        grouped['min'],
        grouped['max'],
        alpha=0.2,
        color='blue',
        label='Min-Max Range'
    )

    # Highlight training count
    ax.axvline(x=TRAIN_N_AGENTS, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Training Count')
    ax.axvspan(TRAIN_N_AGENTS-10, TRAIN_N_AGENTS+10, alpha=0.1, color='red')

    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Final Tracking MSE', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Zero-Shot Transfer Across Agent Counts', fontsize=14)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    plot_path = save_dir / "transfer_scalability.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {plot_path.name}")


def plot_efficiency_metric(results_df, save_dir):
    """
    Plots efficiency metric: MSE normalized by sqrt(n_agents).
    Shows if performance scales with coverage.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute agent spacing (domain units between agents)
    results_df['Agent_Spacing'] = results_df['N_Agents'].apply(
        lambda n: PHYSICS_CONFIG['L_domain'] / jnp.sqrt(n)
    )

    # Compute efficiency (normalized MSE)
    results_df['Efficiency'] = results_df['Final_MSE'] / jnp.sqrt(results_df['N_Agents'])

    # Group by spacing
    grouped = results_df.groupby('Agent_Spacing')['Final_MSE'].agg(['mean', 'std'])

    ax.errorbar(
        grouped.index,
        grouped['mean'],
        yerr=grouped['std'],
        marker='s',
        markersize=8,
        capsize=5,
        color='green'
    )

    ax.set_xlabel('Agent Spacing (domain units)', fontsize=12)
    ax.set_ylabel('Final Tracking MSE', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Performance vs Agent Density', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    plot_path = save_dir / "transfer_efficiency.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {plot_path.name}")


def plot_trajectory_snapshots(params, model, u_init, save_dir):
    """
    Plots trajectory snapshots for different agent counts.
    Shows how control quality degrades/improves with density.
    """
    selected_counts = [36, 100, 196]  # Low, training, high
    time_snapshots = [0, 25, 49]  # Start, middle, end

    fig, axes = plt.subplots(len(selected_counts), len(time_snapshots), figsize=(12, 10))

    key = jax.random.PRNGKey(123)

    for i, n_agents in enumerate(selected_counts):
        key, subkey = jax.random.split(key)
        u_traj, xi_fixed = evaluate_full_trajectory(params, model, u_init, n_agents, subkey)

        for j, t_idx in enumerate(time_snapshots):
            ax = axes[i, j]

            # Plot state
            im = ax.imshow(
                u_traj[t_idx],
                extent=[0, PHYSICS_CONFIG['L_domain'], 0, PHYSICS_CONFIG['L_domain']],
                origin='lower',
                cmap='RdBu_r',
                vmin=-2, vmax=2
            )

            # Overlay agents
            ax.scatter(xi_fixed[:, 0], xi_fixed[:, 1], c='black', s=10, alpha=0.5, marker='x')

            # Labels
            if i == 0:
                ax.set_title(f't={t_idx}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'N={n_agents}\n({"Train" if n_agents==100 else "Transfer"})', fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='u(x, y)')

    plt.suptitle('Trajectory Evolution Across Agent Counts', fontsize=14)

    plot_path = save_dir / "transfer_trajectories.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {plot_path.name}")


def plot_energy_decay(params, model, u_inits, save_dir):
    """
    Plots energy decay over time for different agent counts.
    """
    selected_counts = [36, 64, 100, 144, 196]
    colors = plt.cm.viridis(jnp.linspace(0, 1, len(selected_counts)))

    fig, ax = plt.subplots(figsize=(10, 6))

    key = jax.random.PRNGKey(456)

    for n_agents, color in zip(selected_counts, colors):
        energies = []

        for u_init in u_inits[:5]:  # Use first 5 ICs
            key, subkey = jax.random.split(key)
            u_traj, _ = evaluate_full_trajectory(params, model, u_init, n_agents, subkey)

            # Compute energy over time
            energy_traj = jnp.mean(u_traj**2, axis=(1, 2))
            energies.append(energy_traj)

        # Average over ICs
        mean_energy = jnp.mean(jnp.array(energies), axis=0)

        # Time axis
        time = jnp.arange(PHYSICS_CONFIG['T_steps']) * PHYSICS_CONFIG['substeps'] * PHYSICS_CONFIG['dt']

        label = f'N={n_agents}' + (' (Train)' if n_agents == 100 else '')
        ax.plot(time, mean_energy, label=label, color=color, linewidth=2)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy (Mean Square)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Energy Decay Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    plot_path = save_dir / "transfer_energy_decay.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {plot_path.name}")


def main():
    print("="*70)
    print("KS2D Agent Count Transfer Study - Evaluation")
    print("="*70)

    save_dir = Path("figures/agent_transfer")

    if not save_dir.exists():
        print(f"Error: {save_dir} not found. Run train_transfer.py first.")
        return

    # Load model
    print("\nLoading trained model...")
    params = load_model(save_dir)

    model = DecentralizedKS2DControlNet(
        features=(64, 128),
        domain_size=(PHYSICS_CONFIG['L_domain'], PHYSICS_CONFIG['L_domain']),
        u_max=5.0,
        patch_size=12
    )

    # Load test data
    print("Loading test initial conditions...")
    u_pool = get_or_create_data(
        PHYSICS_CONFIG['N_grid'],
        PHYSICS_CONFIG['L_domain'],
        pool_size=500
    )

    # Sample test ICs
    key = jax.random.PRNGKey(999)
    idx = jax.random.randint(key, (N_TEST_SAMPLES,), 0, u_pool.shape[0])
    u_test = u_pool[idx]

    print(f"Test samples: {N_TEST_SAMPLES}")
    print(f"Test agent counts: {TEST_AGENT_COUNTS}")

    # Evaluate all agent counts
    results = []

    pbar = tqdm(total=len(TEST_AGENT_COUNTS), desc="Evaluating")

    for n_agents in TEST_AGENT_COUNTS:
        xi_fixed = get_agent_grid(n_agents, PHYSICS_CONFIG['L_domain'])
        u_target = jnp.zeros((PHYSICS_CONFIG['N_grid'], PHYSICS_CONFIG['N_grid']))

        for i, u_init in enumerate(u_test):
            key, subkey = jax.random.split(key)

            _, final_mse, energy_decay = evaluate_policy(
                params, model, u_init, xi_fixed, u_target,
                **PHYSICS_CONFIG,
                key=subkey,
                **NOISE_CONFIG
            )

            results.append({
                'N_Agents': n_agents,
                'Sample_ID': i,
                'Final_MSE': float(final_mse),
                'Energy_Decay': float(energy_decay)
            })

        pbar.update(1)

    pbar.close()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_path = save_dir / "transfer_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_scalability_curve(results_df, save_dir)
    plot_efficiency_metric(results_df, save_dir)
    plot_trajectory_snapshots(params, model, u_test[0], save_dir)
    plot_energy_decay(params, model, u_test, save_dir)

    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)

    summary = results_df.groupby('N_Agents')['Final_MSE'].agg(['mean', 'std', 'min', 'max'])
    print(summary)

    # Transfer quality metrics
    print("\n" + "="*70)
    print("Transfer Quality Metrics")
    print("="*70)

    baseline_mse = results_df[results_df['N_Agents'] == TRAIN_N_AGENTS]['Final_MSE'].mean()
    print(f"Baseline MSE (N={TRAIN_N_AGENTS}): {baseline_mse:.6f}")

    for n_agents in TEST_AGENT_COUNTS:
        if n_agents != TRAIN_N_AGENTS:
            transfer_mse = results_df[results_df['N_Agents'] == n_agents]['Final_MSE'].mean()
            ratio = transfer_mse / baseline_mse
            print(f"  N={n_agents:3d}: MSE={transfer_mse:.6f}, Ratio={ratio:.2f}x")

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print(f"Results saved in: {save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
