"""
Sensor Dimension Experiment - Visualization Script (KS-1D)
Evaluates how Sensor Range (float) impacts zero-shot scalability.
Focus: Clean, readable statistical plots only.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import flax.serialization
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.ticker as ticker

# Force CPU to avoid memory fragmentation during small evals
jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

# Output Directory
EXPERIMENT_DIR = Path("figures/sensor_dim")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# --- KS Specific Imports ---
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions
from models.policy_ks1d import DecentralizedControlNet

# --- Config ---
# Matches the training setup
SENSOR_RANGES = [2, 5, 10, 20, 50, 100]
TEST_AGENT_COUNTS = range(20, 55, 5)

# Physics Constants
L_DOMAIN = 22.0     
N_GRID = 128        
T_STEPS = 300
N_TEST_SAMPLES = 50 

def load_params(model, filepath):
    """Safely loads model parameters."""
    if not filepath.exists(): 
        return None
    
    with open(filepath, 'rb') as f: 
        bytes_data = f.read()
    
    # Init dummy params of the correct shape
    # We pass full grid size; the model handles slicing internally
    dummy_init = model.init(
        jax.random.PRNGKey(0), 
        jnp.zeros((N_GRID,)), 
        jnp.zeros((N_GRID,)), 
        jnp.zeros((30,))
    )
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def evaluate_sensor_dims():
    results = []

    # 1. Generate Test Data (Chaotic Spin-up)
    print("Generating KS Chaotic Initial Conditions...")
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    # Get a batch of valid chaotic states
    u_init_test = get_batch_initial_conditions(subkey, N_TEST_SAMPLES, N_GRID, L_DOMAIN)
    u_target_test = jnp.zeros_like(u_init_test) # Target is zero (stability)

    # 2. Loop through Sensor Ranges
    for s_range in SENSOR_RANGES:
        print(f"--- Evaluating Sensor Range: {s_range} ---")
        
        # Instantiate Model with the specific range
        model = DecentralizedControlNet(features=(64, 64), L_domain=L_DOMAIN, window_size=s_range)
        
        # Load weights
        param_path = EXPERIMENT_DIR / f"sensor_dim_{s_range}_params.msgpack"
        params = load_params(model, param_path)
        
        if params is None:
            print(f"   [Skipping] Weights not found: {param_path}")
            continue

        # Setup Dynamics
        dynamics = PDEDynamics(policy_apply_fn=model.apply)

        # 3. Calculate MSE for varying Agent Counts
        print(f"   > Calculating statistics...")
        for n_agents in TEST_AGENT_COUNTS:
            # Create equidistant positions for this agent count
            xi_test = jnp.linspace(0.0, L_DOMAIN, n_agents, endpoint=False) + (L_DOMAIN/n_agents)/2
            xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))

            # JIT-compiled single run
            @jax.jit
            def run_single(u_i, u_t, xi_i):
                # Unroll full trajectory
                u_traj, _, _, _ = dynamics.unroll_controlled(
                    u_i, xi_i, u_t, params, T_STEPS, 
                    N_grid=N_GRID, L=L_DOMAIN, key=jax.random.PRNGKey(0)
                )
                # Return final MSE
                return jnp.mean((u_traj[-1] - u_t)**2) 

            # Vectorize over the test batch
            final_mses = jax.vmap(run_single)(u_init_test, u_target_test, xi_batch)
            
            results.append({
                "Sensor Range": s_range,
                "Agents": n_agents,
                "MSE": float(jnp.mean(final_mses)),
                "Std": float(jnp.std(final_mses))
            })

    return pd.DataFrame(results)

def plot_sensor_sensitivity(df):
    """
    Generates a clean, readable line plot with external legend.
    """
    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Increase figure width to accommodate external legend
    plt.figure(figsize=(10, 6))
    
    # Ensure 'Sensor Range' is treated as categorical/ordinal for proper coloring
    df_plot = df.copy()
    
    # Main Plot
    sns.lineplot(
        data=df_plot, 
        x="Agents", 
        y="MSE", 
        hue="Sensor Range", 
        palette="viridis",      # High contrast sequential colormap
        style="Sensor Range",   # Different dashes help distinguish lines
        markers=True, 
        markersize=8, 
        linewidth=2.5,
        dashes=False            # Optional: Turn off dashes if too messy, or keep default
    )
    
    # Add vertical line for Training Distribution
    plt.axvline(x=30, color='red', linestyle='--', alpha=0.6, label="Training Density (N=30)")
    
    # Labels and Titles
    plt.title("Sensor Range Sensitivity (KS-1D)", fontsize=16, pad=15)
    plt.ylabel("Final Tracking Error (MSE)", fontsize=12)
    plt.xlabel("Deployment Agent Count", fontsize=12)
    
    # Log Scale for MSE
    plt.yscale('log')
    
    # Improve Grid Readability for Log Scale
    ax = plt.gca()
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    
    # CLEAN LEGEND PLACEMENT
    # Move legend outside the plot to the right
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        borderaxespad=0.,
        title="Sensor Range",
        fontsize=11,
        title_fontsize=12
    )
    
    plt.tight_layout()
    
    save_path = EXPERIMENT_DIR / "sensor_dimension_sensitivity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    print(f"--- Starting KS Sensor Dimension Analysis ---")
    
    df_results = evaluate_sensor_dims()
    
    if not df_results.empty:
        # Save raw data
        csv_path = EXPERIMENT_DIR / "sensor_metrics.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
        
        # Plot
        plot_sensor_sensitivity(df_results)
    else:
        print("No results generated. Check if model parameter files exist in 'figures/sensor_dim'.")