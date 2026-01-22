"""
Phase 2: Noise Robustness Study - Visualization

Evaluates all 7 trained models on 9 test scenarios across 5 agent counts.
Generates:
1. Scalability curves (9 plots) - one per test scenario
2. Cross-robustness heatmaps
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

# Test Scenarios (9 total)
TEST_SCENARIOS = {
    "State_Low":  {"u": 0.0, "z": 0.025},
    "State_Mid":  {"u": 0.0, "z": 0.1},
    "State_High": {"u": 0.0, "z": 0.5},

    "Actuator_Low":  {"u": 0.05, "z": 0.0},
    "Actuator_Mid":  {"u": 0.2,  "z": 0.0},
    "Actuator_High": {"u": 1.0,  "z": 0.0},

    "Combined_Low":  {"u": 0.05, "z": 0.025},
    "Combined_Mid":  {"u": 0.2,  "z": 0.1},
    "Combined_High": {"u": 1.0,  "z": 0.5},
}

# Model Names (7 total)
MODEL_NAMES = [
    "baseline_clean",
    "actuator_low", "actuator_mid", "actuator_high",
    "state_low", "state_mid", "state_high"
]

# Test Parameters
TEST_AGENT_COUNTS = [60, 80, 100, 120, 140]  # Include training count (100)
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


def load_model(model_name, save_dir):
    """Load trained parameters for a model."""
    params_path = save_dir / f"{model_name}_params.msgpack"
    if not params_path.exists():
        raise FileNotFoundError(f"Model not found: {params_path}")

    with open(params_path, 'rb') as f:
        params = flax.serialization.from_bytes(None, f.read())

    return params


def evaluate_scenario(params, model, u_inits, scenario_name, noise_config, n_agents):
    """
    Evaluates a model on a scenario across multiple initial conditions.

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
            noise_u=noise_config['u'],
            noise_z=noise_config['z']
        )

        mse_values.append(final_mse)
        energy_values.append(energy_decay)

    return jnp.array(mse_values), jnp.array(energy_values)


def plot_scalability_curve(results_df, scenario_name, save_dir):
    """
    Plots scalability curves for a single scenario.
    X-axis: Agent count
    Y-axis: MSE (log scale)
    Lines: Different trained models
    """
    plt.figure(figsize=(10, 6))

    scenario_data = results_df[results_df['Scenario'] == scenario_name]

    for model_name in MODEL_NAMES:
        model_data = scenario_data[scenario_data['Model'] == model_name]

        # Group by agent count and compute mean/std
        grouped = model_data.groupby('N_Agents')['Final_MSE'].agg(['mean', 'std'])

        plt.errorbar(
            grouped.index,
            grouped['mean'],
            yerr=grouped['std'],
            marker='o',
            label=model_name,
            capsize=5,
            alpha=0.8
        )

    plt.axvline(x=TRAIN_N_AGENTS, color='red', linestyle='--', alpha=0.5, label='Training Count')
    plt.yscale('log')
    plt.xlabel('Number of Agents')
    plt.ylabel('Final Tracking MSE')
    plt.title(f'Scalability - {scenario_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    plot_path = save_dir / f"scalability_{scenario_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {plot_path.name}")


def plot_cross_robustness_heatmap(results_df, save_dir):
    """
    Creates heatmap showing cross-robustness:
    Rows: Models
    Cols: Test scenarios
    Color: Mean MSE (normalized)
    """
    # Aggregate data: mean MSE across all agent counts and samples
    pivot_data = results_df.groupby(['Model', 'Scenario'])['Final_MSE'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='Model', columns='Scenario', values='Final_MSE')

    # Reorder for better visualization
    model_order = MODEL_NAMES
    scenario_order = list(TEST_SCENARIOS.keys())

    pivot_table = pivot_table.reindex(index=model_order, columns=scenario_order)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(range(len(scenario_order)))
    ax.set_yticks(range(len(model_order)))
    ax.set_xticklabels(scenario_order, rotation=45, ha='right')
    ax.set_yticklabels(model_order)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Final MSE', rotation=270, labelpad=20)

    # Annotate cells with values
    for i in range(len(model_order)):
        for j in range(len(scenario_order)):
            text = ax.text(j, i, f'{pivot_table.values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Cross-Robustness Matrix (Lower is Better)')
    ax.set_xlabel('Test Scenario')
    ax.set_ylabel('Trained Model')
    plt.tight_layout()

    heatmap_path = save_dir / "cross_robustness_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {heatmap_path.name}")


def main():
    print("="*70)
    print("KS2D Noise Robustness Study - Evaluation")
    print("="*70)

    save_dir = Path("figures/noise_experiments/decoupled_robustness")

    if not save_dir.exists():
        print(f"Error: {save_dir} not found. Run train_noise_decoupled.py first.")
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
    print(f"Test scenarios: {len(TEST_SCENARIOS)}")
    print(f"Total evaluations: {len(MODEL_NAMES)} models × {len(TEST_SCENARIOS)} scenarios × {len(TEST_AGENT_COUNTS)} counts × {N_TEST_SAMPLES} samples")

    # Initialize model architecture (will load different params)
    model = DecentralizedKS2DControlNet(
        features=(64, 128),
        domain_size=(PHYSICS_CONFIG['L_domain'], PHYSICS_CONFIG['L_domain']),
        u_max=5.0,
        patch_size=12
    )

    # Collect results
    results = []

    # Evaluate all combinations
    total_evals = len(MODEL_NAMES) * len(TEST_SCENARIOS) * len(TEST_AGENT_COUNTS)
    pbar = tqdm(total=total_evals, desc="Evaluating")

    for model_name in MODEL_NAMES:
        # Load model parameters
        params = load_model(model_name, save_dir)

        for scenario_name, noise_config in TEST_SCENARIOS.items():
            for n_agents in TEST_AGENT_COUNTS:
                # Evaluate on this configuration
                mse_vals, energy_vals = evaluate_scenario(
                    params, model, u_test, scenario_name, noise_config, n_agents
                )

                # Store results (one row per sample)
                for i in range(N_TEST_SAMPLES):
                    results.append({
                        'Model': model_name,
                        'Scenario': scenario_name,
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
    csv_path = save_dir / "noise_robustness_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")

    # Generate plots
    print("\nGenerating scalability curves...")
    for scenario_name in TEST_SCENARIOS.keys():
        plot_scalability_curve(results_df, scenario_name, save_dir)

    print("\nGenerating cross-robustness heatmap...")
    plot_cross_robustness_heatmap(results_df, save_dir)

    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)

    summary = results_df.groupby(['Model', 'Scenario'])['Final_MSE'].agg(['mean', 'std'])
    print(summary)

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print(f"Results saved in: {save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
