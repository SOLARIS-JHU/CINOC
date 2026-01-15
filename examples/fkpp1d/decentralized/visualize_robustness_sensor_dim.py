"""
Sensor Dimension Experiment - Visualization Script (1D)
Evaluates how Sensor Range (float) impacts zero-shot scalability.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from tesseract_core import Tesseract
import sys
import flax.serialization
from flax import linen as nn
from typing import Sequence
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.ticker as ticker

jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent

sys.path.append(str(script_dir.parent.parent.parent))

# Output Directory (matches the runner)
EXPERIMENT_DIR = Path("figures/sensor_dim")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

from dynamics_dual import PDEDynamics 
from data_utils import generate_grf
from models.policy import DecentralizedControlNet

# --- Config ---
# The variable we are testing 
SENSOR_RANGES = [0.04, 0.06, 0.08, 0.12, 0.2, 0.3]

TEST_AGENT_COUNTS = range(20, 101, 20)
N_PDE = 100
T_STEPS = 300
N_TEST_SAMPLES = 50 

def load_params(model, filepath):
    """Initializes the model with weights from file."""
    if not filepath.exists():
        return None
        
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
    
    dummy_init = model.init(
        jax.random.PRNGKey(0), 
        jnp.zeros((N_PDE,)), 
        jnp.zeros((N_PDE,)), 
        jnp.zeros((30,)) # 1D positions: shape (30,)
    )
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def evaluate_sensor_dims():
    solver_ts = Tesseract.from_image("solver_fkpp1d_decentralized:latest")
    results = []

    # Pre-generate Test Data
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    _, z_init_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.2))(jax.random.split(k1, N_TEST_SAMPLES))
    _, z_target_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.4))(jax.random.split(k2, N_TEST_SAMPLES))

    with solver_ts:
        # Loop through Sensor Ranges
        for s_range in SENSOR_RANGES:
            print(f"--- Evaluating Sensor Range: {s_range} ---")
            
            # 1. Instantiate Model with SPECIFIC sensor_range
            model = DecentralizedControlNet(
                features=(64, 64), 
                sensor_range=s_range
            )
            
            # 2. Load Weights (Match filename format from runner: 0.04 -> 004)
            # Replaces dot with empty string for filename
            # clean_name = str(s_range).replace('.', '')
            param_path = EXPERIMENT_DIR / f"sensor_dim_{s_range}_params.msgpack"
            
            params = load_params(model, param_path)
            
            if params is None:
                print(f"Skipping range {s_range} (Weights not found at {param_path})")
                continue

            dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)

            for n_agents in TEST_AGENT_COUNTS:
                print(f"   > Agents: {n_agents}")
                
                # Create 1D agent positions
                xi_test = jnp.linspace(0.1, 0.9, n_agents)
                xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))

                @jax.jit
                def run_single(z_i, z_t, xi_i):
                    z_traj, _, _, _ = dynamics.unroll_controlled(
                        z_i, xi_i, z_t, params, T_STEPS, 
                        key=jax.random.PRNGKey(0), 
                        noise_u=0, 
                        noise_z=0
                    )
                    return jnp.mean((z_traj[-1] - z_t)**2) 

                final_mses = jax.vmap(run_single)(z_init_test, z_target_test, xi_batch)
                
                avg_mse = float(jnp.mean(final_mses))
                std_mse = float(jnp.std(final_mses))

                results.append({
                    "Sensor Range": s_range,
                    "Agents": n_agents,
                    "MSE": avg_mse,
                    "Std": std_mse
                })

    return pd.DataFrame(results)

def plot_sensor_sensitivity(df):
    # Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    
    # 1. Adapt Colors: 
    # specific distinct colors (tab10 is high contrast)
    # We force 'Sensor Range' to string/category so seaborn picks distinct colors 
    # instead of a gradient.
    df_plot = df.copy()
    df_plot["Sensor Range"] = df_plot["Sensor Range"].astype(str)
    
    # Sort strings so the legend is in numeric order (0.04, 0.06...) not (0.04, 0.1, 0.06)
    unique_ranges = sorted(df["Sensor Range"].unique())
    unique_ranges_str = [str(x) for x in unique_ranges]

    sns.lineplot(
        data=df_plot, 
        x="Agents", 
        y="MSE", 
        hue="Sensor Range", 
        hue_order=unique_ranges_str, # Force numeric ordering
        style="Sensor Range",
        palette="tab10", 
        markers=True, 
        markersize=9, 
        linewidth=2.5
    )
    
    # Vertical line for training scale
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training Scale (N=30)")
    
    # Labels
    plt.title(f"Sensor Range Sensitivity", fontsize=16)
    plt.ylabel("Final Tracking Error (MSE)", fontsize=12)
    plt.xlabel("Deployment Agent Count", fontsize=12)
    
    # 2. Adapt Y-Axis Numbers:
    plt.yscale('log')
    ax = plt.gca()
    
    # This forces ticks at [1, 2, 3, 4, 5...] x 10^x
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[2.0, 5.0], numticks=15))
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.1e"))
    
    # Turn on the minor grid so you can see the subdivisions
    ax.grid(True, which="minor", ls=":", alpha=0.4)
    

    # 3. Adapt Legend:
    # loc='upper right' puts it inside. 
    # framealpha=0.9 gives it a background so lines don't strike through the text.
    plt.legend(
        title="Sensor Range (%)", 
        loc='upper right', 
        frameon=True, 
        framealpha=0.95,
        edgecolor='white'
    )
    
    save_path = EXPERIMENT_DIR / "sensor_dimension_sensitivity.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved sensitivity plot to {save_path}")

if __name__ == "__main__":
    print(f"Starting Sensor Dimension Visualization...")
    print(f"Reading from: {EXPERIMENT_DIR}")
    
    df_results = evaluate_sensor_dims()
    
    if not df_results.empty:
        df_results.to_csv(EXPERIMENT_DIR / "sensor_metrics.csv", index=False)
        print("Metrics saved to CSV.")
        plot_sensor_sensitivity(df_results)
    else:
        print("No results generated. Check if model parameters exist.")