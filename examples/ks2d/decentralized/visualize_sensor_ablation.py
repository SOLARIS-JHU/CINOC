"""
Phase 3: Sensor Dimension Ablation - Visualization

Evaluates models with different patch sizes across various agent counts.
Generates:
1. Scalability curves (X=agent count, Y=MSE, Lines=patch sizes)
2. Relative performance heatmap
3. CSV metrics for analysis
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

# Patch sizes (must match training)
PATCH_SIZES = [6, 8, 12, 16, 20, 24]

# Test Parameters
TEST_AGENT_COUNTS = [49, 64, 81, 100, 121, 144, 169]  # 7²...13² (training=100)
N_TEST_SAMPLES = 50
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

# Noise (fixed low values for clean comparison)
NOISE_CONFIG = {
    'noise_u': 0.05,
    'noise_z': 0.025,
}


def load_model(patch_size, save_dir):
    """Load trained parameters for a model."""
    params_path = save_dir / f"patch_{patch_size}_params.msgpack"
    if not params_path.exists():
        raise FileNotFoundError(f"Model not found: {params_path}")

    with open(params_path, 'rb') as f:
        params = flax.serialization.from_bytes(None, f.read())

    return params


def evaluate_configuration(params, model, u_inits, n_agents):
    """
    Evaluates a model on a configuration across multiple initial conditions.

    Returns:
        mse_values: Array of final MSE for each IC
        energy_decay_values: Array of energy decay rates
    """
    xi_fixed = get_agent_grid(n_agents, PHYSICS_CONFIG['L_domain'])
    u_target = jnp.zeros((PHYSICS_CONFIG['N_grid'], PHYSICS_CONFIG['N_grid']))

    mse_values = []
    energy_values = []

    key = jax.random.PRNGKey(42)

    for u_init in u_inits:
        key, subkey = jax.random.split(key)

        _, final_mse, energy_decay = evaluate_policy(
            params, model, u_init, xi_fixed, u_target,
            **PHYSICS_CONFIG,
            key=subkey,
            **NOISE_CONFIG
        )

        mse_values.append(final_mse)
        energy_values.append(energy_decay)

    return jnp.array(mse_values), jnp.array(energy_values)


def plot_scalability_curves(results_df, save_dir):
    """
    Plots scalability curves:
    X-axis: Agent count
    Y-axis: MSE (log scale)
    Lines: Different patch sizes
    """
    plt.figure(figsize=(12, 7))

    for patch_size in PATCH_SIZES:
        patch_data = results_df[results_df['Patch_Size'] == patch_size]

        # Group by agent count and compute mean/std
        grouped = patch_data.groupby('N_Agents')['Final_MSE'].agg(['mean', 'std'])

        plt.errorbar(
            grouped.index,
            grouped['mean'],
            yerr=grouped['std'],
            marker='o',
            label=f'{patch_size}px',
            capsize=5,
            alpha=0.8
        )

    plt.axvline(x=TRAIN_N_AGENTS, color='red', linestyle='--', alpha=0.5, label='Training Count')
    plt.yscale('log')
    plt.xlabel('Number of Agents')
    plt.ylabel('Final Tracking MSE')
    plt.title('Sensor Dimension Ablation: Zero-Shot Scalability')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    plot_path = save_dir / "sensor_scalability_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {plot_path.name}")


def plot_relative_performance_heatmap(results_df, save_dir):
    """
    Creates heatmap showing relative performance:
    Rows: Patch sizes
    Cols: Agent counts
    Color: MSE normalized by baseline (12px)
    """
    # Aggregate data: mean MSE across samples
    pivot_data = results_df.groupby(['Patch_Size', 'N_Agents'])['Final_MSE'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='Patch_Size', columns='N_Agents', values='Final_MSE')

    # Normalize by baseline (12px)
    baseline_row = pivot_table.loc[12]
    normalized_table = pivot_table.div(baseline_row, axis=1)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(normalized_table.values, cmap='RdYlGn_r', aspect='auto', vmin=0.5, vmax=2.0)

    # Labels
    ax.set_xticks(range(len(TEST_AGENT_COUNTS)))
    ax.set_yticks(range(len(PATCH_SIZES)))
    ax.set_xticklabels(TEST_AGENT_COUNTS)
    ax.set_yticklabels(PATCH_SIZES)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative MSE (vs 12px baseline)', rotation=270, labelpad=20)

    # Annotate cells with values
    for i in range(len(PATCH_SIZES)):
        for j in range(len(TEST_AGENT_COUNTS)):
            value = normalized_table.values[i, j]
            color = "white" if value > 1.2 or value < 0.8 else "black"
            text = ax.text(j, i, f'{value:.2f}',
                          ha="center", va="center", color=color, fontsize=9)

    # Highlight training count
    train_idx = TEST_AGENT_COUNTS.index(TRAIN_N_AGENTS)
    ax.axvline(x=train_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_title('Relative Performance (Green=Better, Red=Worse)')
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Patch Size (pixels)')
    plt.tight_layout()

    heatmap_path = save_dir / "sensor_relative_performance.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {heatmap_path.name}")


def plot_agent_spacing_analysis(results_df, save_dir):
    """
    Plots MSE vs agent spacing (domain units between agents).
    Shows which patch sizes match which densities.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert agent counts to spacing
    results_df['Agent_Spacing'] = results_df['N_Agents'].apply(
        lambda n: PHYSICS_CONFIG['L_domain'] / jnp.sqrt(n)
    )

    for patch_size in PATCH_SIZES:
        patch_data = results_df[results_df['Patch_Size'] == patch_size]

        # Group by spacing
        grouped = patch_data.groupby('Agent_Spacing')['Final_MSE'].agg(['mean', 'std'])

        # Compute receptive field in domain units
        receptive_field = patch_size * PHYSICS_CONFIG['L_domain'] / PHYSICS_CONFIG['N_grid']

        ax.errorbar(
            grouped.index,
            grouped['mean'],
            yerr=grouped['std'],
            marker='o',
            label=f'{patch_size}px (RF={receptive_field:.1f})',
            capsize=5,
            alpha=0.8
        )

    ax.set_xlabel('Agent Spacing (domain units)')
    ax.set_ylabel('Final Tracking MSE')
    ax.set_yscale('log')
    ax.set_title('Performance vs Agent Spacing')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    spacing_path = save_dir / "sensor_spacing_analysis.png"
    plt.savefig(spacing_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {spacing_path.name}")


def main():
    print("="*70)
    print("KS2D Sensor Dimension Ablation - Evaluation")
    print("="*70)

    save_dir = Path("figures/sensor_ablation")

    if not save_dir.exists():
        print(f"Error: {save_dir} not found. Run train_sensor_ablation.py first.")
        return

    # Load test data
    print("\nLoading test initial conditions...")
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
    print(f"Total evaluations: {len(PATCH_SIZES)} patches × {len(TEST_AGENT_COUNTS)} counts × {N_TEST_SAMPLES} samples")

    # Collect results
    results = []

    # Evaluate all combinations
    total_evals = len(PATCH_SIZES) * len(TEST_AGENT_COUNTS)
    pbar = tqdm(total=total_evals, desc="Evaluating")

    for patch_size in PATCH_SIZES:
        # Load model parameters
        params = load_model(patch_size, save_dir)

        # Create model with correct patch size
        model = DecentralizedKS2DControlNet(
            features=(64, 128),
            domain_size=(PHYSICS_CONFIG['L_domain'], PHYSICS_CONFIG['L_domain']),
            u_max=5.0,
            patch_size=patch_size
        )

        for n_agents in TEST_AGENT_COUNTS:
            # Evaluate on this configuration
            mse_vals, energy_vals = evaluate_configuration(
                params, model, u_test, n_agents
            )

            # Store results (one row per sample)
            for i in range(N_TEST_SAMPLES):
                results.append({
                    'Patch_Size': patch_size,
                    'N_Agents': n_agents,
                    'Sample_ID': i,
                    'Final_MSE': float(mse_vals[i]),
                    'Energy_Decay': float(energy_vals[i])
                })

            pbar.update(1)

    pbar.close()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_path = save_dir / "sensor_ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_scalability_curves(results_df, save_dir)
    plot_relative_performance_heatmap(results_df, save_dir)
    plot_agent_spacing_analysis(results_df, save_dir)

    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)

    summary = results_df.groupby(['Patch_Size', 'N_Agents'])['Final_MSE'].agg(['mean', 'std'])
    print(summary)

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print(f"Results saved in: {save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
