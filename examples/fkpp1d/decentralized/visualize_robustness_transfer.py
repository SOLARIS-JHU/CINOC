"""
Robustness Transfer Experiment - Comprehensive Decoupled Visualization
Generates 8 Scalability Plots covering the full matrix of noise scenarios.
Compares Actuator-Trained vs. State-Trained models across all environments.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from tesseract_core import Tesseract
import sys
import flax.serialization
from pathlib import Path
from functools import partial
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

EXPERIMENT_DIR = Path("figures/noise_experiments/decoupled_robustness")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

# --- Config ---
# We test on a spread of agent counts to show the curve
TEST_AGENT_COUNTS = [20, 30, 40, 60, 80, 100]
N_PDE = 100
T_STEPS = 300
N_TEST_SAMPLES = 20  # Reduced slightly for speed across 8 scenarios

# --- Model Definitions ---
# We will load ALL these models to compare them in every plot
ALL_MODELS = [
    "baseline_clean",
    # Actuator Specialists
    "actuator_only_0p02",
    "actuator_only_0p1",
    # "actuator_only_0p5",
    # State Specialists
    "state_only_0p01",
    "state_only_0p05",
    # "state_only_0p25"
]

# --- Scenario Definitions ---
# The 8 scenarios requested by the user
SCENARIOS = {
    # 1-3: State Noise Only (Low, Mid, High)
    "State_Low":  {"u": 0.0, "z": 0.01},
    "State_Mid":  {"u": 0.0, "z": 0.05},
    "State_High": {"u": 0.0, "z": 0.25},
    
    # 4-6: Actuator Noise Only (Low, Mid, High)
    # (Assuming "Sensor Noise" in prompt meant Actuator to cover the decoupled axis)
    "Actuator_Low":  {"u": 0.02, "z": 0.0},
    "Actuator_Mid":  {"u": 0.1,  "z": 0.0},
    "Actuator_High": {"u": 0.5,  "z": 0.0},
    
    # 7-9: Combined
    "Combined_Low":  {"u": 0.02, "z": 0.01},
    "Combined_Mid": {"u": 0.1,  "z": 0.05},
    "Combined_High": {"u": 0.5,  "z": 0.25},
}


def load_params(model, filepath):
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
    dummy_init = model.init(jax.random.PRNGKey(0), jnp.zeros((N_PDE,)), jnp.zeros((N_PDE,)), jnp.zeros((30,)))
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def evaluate_scenario(scenario_name, noise_u, noise_z, loaded_params, dynamics):
    """
    Evaluates all loaded models on a single scenario.
    """
    print(f"\n--- Evaluating Scenario: {scenario_name} (u={noise_u}, z={noise_z}) ---")
    
    # Generate Test Data (Fixed Seed per scenario for fairness)
    key = jax.random.PRNGKey(42)
    key, k1, k2 = jax.random.split(key, 3)
    _, z_init_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.2))(jax.random.split(k1, N_TEST_SAMPLES))
    _, z_target_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.4))(jax.random.split(k2, N_TEST_SAMPLES))

    results = []

    for n_agents in TEST_AGENT_COUNTS:
        sys.stdout.write(f"\rAgents: {n_agents}...")
        sys.stdout.flush()
        
        xi_test = jnp.linspace(0.1, 0.9, n_agents)
        xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))
        
        for m_name, params in loaded_params.items():
            
            # JIT Unroll wrapper
            def run_single(z_i, z_t, xi_i):
                z_traj, _, _, _ = dynamics.unroll_controlled(
                    z_i, xi_i, z_t, params, T_STEPS, 
                    key=jax.random.PRNGKey(0), 
                    noise_u=noise_u, 
                    noise_z=noise_z
                )
                return jnp.mean((z_traj[-1] - z_t)**2) 
            
            final_mses = jax.vmap(run_single)(z_init_test, z_target_test, xi_batch)
            
            results.append({
                "Model": m_name,
                "Agents": n_agents,
                "MSE": float(jnp.mean(final_mses)),
                "Std": float(jnp.std(final_mses)),
                "Scenario": scenario_name
            })
    print(" Done.")
    return pd.DataFrame(results)

def plot_comprehensive(df, title, filename):
    """
    Plots all models with distinct color coding:
    - Baseline: Black/Grey
    - Actuator Models: Reds
    - State Models: Blues
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # 1. Prettify Labels
    def prettify(name):
        if "baseline" in name: return "Baseline"
        clean = name.replace("actuator_only", "Actuator").replace("state_only", "State")
        clean = clean.replace("_", " ")
        clean = clean.replace("p", ".")
        # e.g. "Actuator 0.5"
        return clean

    df['Label'] = df['Model'].apply(prettify)
    
    # 2. Assign Colors manually
    unique_labels = sorted(df['Label'].unique())
    palette = {}
    
    # Color definitions
    actuator_colors = sns.color_palette("Reds", n_colors=4)[1:] # Skip lightest
    state_colors = sns.color_palette("Blues", n_colors=4)[1:]   # Skip lightest
    
    a_idx, s_idx = 0, 0
    
    for label in unique_labels:
        if "Baseline" in label:
            palette[label] = "#333333" # Dark Grey
        elif "Actuator" in label:
            # Assign increasing red intensity
            palette[label] = actuator_colors[a_idx % len(actuator_colors)]
            a_idx += 1
        elif "State" in label:
            # Assign increasing blue intensity
            palette[label] = state_colors[s_idx % len(state_colors)]
            s_idx += 1
        else:
            palette[label] = "gray"

    # 3. Plot
    sns.lineplot(
        data=df, 
        x="Agents", 
        y="MSE", 
        hue="Label", 
        style="Label", 
        markers=True, 
        markersize=8, 
        linewidth=2.0,
        palette=palette
    )
    
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training Scale (N=30)")
    plt.title(title, fontsize=14)
    plt.ylabel("Final Tracking Error (MSE)", fontsize=12)
    plt.xlabel("Deployment Agent Count", fontsize=12)
    plt.yscale('log')
    plt.legend(title="Policy Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_path = EXPERIMENT_DIR / filename
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

def run_all_scenarios():
    print("Loading Models...")
    solver_ts = Tesseract.from_image("solver_fkpp1d_decentralized:latest")
    
    with solver_ts:
        model = DecentralizedControlNet(features=(64, 64))
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)
        
        # Load all params once
        loaded_params = {}
        for m_name in ALL_MODELS:
            p_path = EXPERIMENT_DIR / f"{m_name}_params_0.001.msgpack"
            if not p_path.exists(): p_path = EXPERIMENT_DIR / f"{m_name}_params"
            
            if p_path.exists():
                loaded_params[m_name] = load_params(model, p_path)
            else:
                print(f"Skipping {m_name} (not found)")

        if not loaded_params:
            print("No models found. Run the training runner first!")
            return

        # Run Scenarios
        for sc_name, env_cfg in SCENARIOS.items():
            df_res = evaluate_scenario(
                sc_name, 
                env_cfg["u"], 
                env_cfg["z"], 
                loaded_params, 
                dynamics
            )
            
            # Save Data
            df_res.to_csv(EXPERIMENT_DIR / f"metrics_{sc_name}.csv", index=False)
            
            # Generate Plot
            title_str = f"Robustness in {sc_name.replace('_', ' ')} Env ($\sigma_u={env_cfg['u']}, \sigma_z={env_cfg['z']}$)"
            file_str = f"plot_robustness_{sc_name}_0.001.pdf"
            
            plot_comprehensive(df_res, title_str, file_str)

if __name__ == "__main__":
    run_all_scenarios()
    print("\nAll comprehensive visualizations completed.")