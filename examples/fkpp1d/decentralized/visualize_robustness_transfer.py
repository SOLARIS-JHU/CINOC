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
TEST_AGENT_COUNTS = [20, 30, 40, 60, 80, 100]
N_PDE = 100
T_STEPS = 300
N_TEST_SAMPLES = 20 

# --- Model Definitions ---
ALL_MODELS = [
    "baseline_clean",
    "actuator_only_0p02",
    "actuator_only_0p1",
    "state_only_0p01",
    "state_only_0p05",
]

# --- Scenario Definitions ---
SCENARIOS = {
    # 1-3: State Noise Only
    "State_Low":  {"u": 0.0, "z": 0.01},
    "State_Mid":  {"u": 0.0, "z": 0.05},
    "State_High": {"u": 0.0, "z": 0.25},
    
    # 4-6: Actuator Noise Only
    "Actuator_Low":  {"u": 0.02, "z": 0.0},
    "Actuator_Mid":  {"u": 0.1,  "z": 0.0},
    "Actuator_High": {"u": 0.5,  "z": 0.0},
    
    # 7-9: Combined
    "Combined_Low":  {"u": 0.02, "z": 0.01},
    "Combined_Mid": {"u": 0.1,  "z": 0.05},
    "Combined_High": {"u": 0.5,  "z": 0.25},
}

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font settings - Times New Roman for papers
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        
        # Font sizes
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        
        # Line widths
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        
        # Spines
        "axes.spines.top": True,
        "axes.spines.right": True,
        
        # Output
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

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
    Plots all models with distinct color coding using Paper Style.
    """
    setup_paper_style()
    
    # Use standard single-column size (approx 5.0 x 3.5 inches)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    
    # 1. Prettify Labels
    def prettify(name):
        if "baseline" in name: return "Baseline"
        clean = name.replace("actuator_only", "Actuator").replace("state_only", "State")
        clean = clean.replace("_", " ")
        clean = clean.replace("p", ".")
        return clean

    df['Label'] = df['Model'].apply(prettify)
    
    # 2. Assign Colors manually
    unique_labels = sorted(df['Label'].unique())
    palette = {}
    
    actuator_colors = sns.color_palette("Reds", n_colors=4)[1:] 
    state_colors = sns.color_palette("Blues", n_colors=4)[1:]   
    
    a_idx, s_idx = 0, 0
    
    for label in unique_labels:
        if "Baseline" in label:
            palette[label] = "#333333" # Dark Grey
        elif "Actuator" in label:
            palette[label] = actuator_colors[a_idx % len(actuator_colors)]
            a_idx += 1
        elif "State" in label:
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
        markersize=6, 
        linewidth=1.5,
        palette=palette,
        ax=ax
    )
    
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training (N=30)")
    ax.set_title(title)
    ax.set_ylabel("Final Tracking Error (MSE)")
    ax.set_xlabel("Deployment Agent Count")
    ax.set_yscale('log')
    
    # Custom grid
    ax.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Legend handling: Put outside to prevent clutter in small plot
    ax.legend(
        title="Policy Type", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        framealpha=0.9,
        fontsize=9
    )
    
    save_path = EXPERIMENT_DIR / filename
    plt.savefig(save_path) # dpi and bbox handled by rcParams
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
            # Try loading specific epoch or default folder
            p_path = EXPERIMENT_DIR / f"{m_name}_params_0.001.msgpack"
            if not p_path.exists(): 
                p_path = EXPERIMENT_DIR / f"{m_name}_params"
            
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
            # Latex formatting for title
            title_str = f"Robustness: {sc_name.replace('_', ' ')} ($\\sigma_u={env_cfg['u']}, \\sigma_z={env_cfg['z']}$)"
            file_str = f"plot_robustness_{sc_name}_0.001.pdf"
            
            plot_comprehensive(df_res, title_str, file_str)

if __name__ == "__main__":
    run_all_scenarios()
    print("\nAll comprehensive visualizations completed.")