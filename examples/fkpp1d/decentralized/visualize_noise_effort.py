import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import flax.serialization
from pathlib import Path
from tesseract_core import Tesseract
from functools import partial

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

# --- Config ---
MODELS_DIR = Path("figures/noise_experiments/robustness_transfer")
OUTPUT_DIR = Path("figures/noise_experiments/effort")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PDE = 100
T_STEPS = 300
TEST_AGENT_COUNTS = list(range(20, 201, 10)) # [20, 30, ..., 100]

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font settings - Times New Roman for papers
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Math font compatible with Times
        
        # Font sizes for two-column paper
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        
        # Line widths
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        
        # Remove top/right spines for cleaner look
        "axes.spines.top": True,
        "axes.spines.right": True,
        
        # High-quality output
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

def load_params(model, filepath):
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
    # Initialize dummy params with the training agent count (30) to match structure
    dummy_init = model.init(jax.random.PRNGKey(0), jnp.zeros((N_PDE,)), jnp.zeros((N_PDE,)), jnp.zeros((30,)))
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def evaluate_effort_scaling(solver_ts):
    """
    Evaluates effort metrics (L1 and L2 norms of control) across varying agent counts.
    Uses the updated API requiring key and noise arguments.
    Considers only the last 70% of the simulation for steady-state analysis.
    """
    results = []
    
    model = DecentralizedControlNet(features=(64, 64))
    dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)
    
    # Define models to test
    model_files = {
        "Baseline": MODELS_DIR / "baseline_params.msgpack",
        "Low Noise": MODELS_DIR / "low_noise_params.msgpack"
    }

    # Load parameters
    loaded_params = {}
    for name, path in model_files.items():
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
        loaded_params[name] = load_params(model, path)
    
    # Generate a fixed evaluation environment
    key = jax.random.PRNGKey(42)
    key_init, key_target = jax.random.split(key)
    # Using a single representative sample for clear trend lines
    _, z_init = generate_grf(key_init, n_points=N_PDE, length_scale=0.2)
    _, z_target = generate_grf(key_target, n_points=N_PDE, length_scale=0.4)

    # Determine window for steady state (last 70%)
    start_step = int(T_STEPS * (1.0 - 0.70)) # Skip first 30%
    print(f"Analyzing metrics from step {start_step} to {T_STEPS} (Steady State)")

    for n_agents in TEST_AGENT_COUNTS:
        print(f"Evaluating N={n_agents}...")
        
        # Interpolate positions for this agent count
        xi_init = jnp.linspace(0.2, 0.8, n_agents)

        for m_name, params in loaded_params.items():
            # Updated API Call: passing key, noise_u, and noise_z
            z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
                z_init, xi_init, z_target, params, T_STEPS, 
                key=jax.random.PRNGKey(0), 
                noise_u=0.0, # Evaluating pure effort scaling without added noise
                noise_z=0.0
            )
            
            # --- Metrics (Windowed) ---
            # u_traj shape: (T_STEPS, n_agents)
            u_steady = u_traj[start_step:, :]
            
            # 1. Total Squared Effort (Energy): mean over time of sum(u^2) over agents
            effort_sq = jnp.mean(jnp.sum(u_steady**2, axis=1))
            
            # 2. Total Absolute Effort (Fuel): mean over time of sum(|u|) over agents
            effort_abs = jnp.mean(jnp.sum(jnp.abs(u_steady), axis=1))

            results.append({
                "Model": m_name,
                "Agents": n_agents,
                "Sum_Sq": float(effort_sq),
                "Sum_Abs": float(effort_abs)
            })

    return pd.DataFrame(results)

def plot_effort_metrics(df):
    # Apply paper style settings
    setup_paper_style()
    
    # Create figure (sized for single-column paper usage, approx 5-6 inches wide)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Consistent styling
    palette = {"Baseline": "#2c3e50", "Low Noise": "#e74c3c"}
    markers = {"Baseline": "o", "Low Noise": "s"}
    
    # --- Quadratic Effort Only ---
    sns.lineplot(
        data=df, x="Agents", y="Sum_Sq", hue="Model", style="Model",
        markers=markers, palette=palette, linewidth=1.5, ax=ax, markersize=7
    )
    
    # Set titles and labels (Font sizes are handled by rcParams now)
    ax.set_title(r"Total Quadratic Effort ($\sum u_i^2$)", fontweight='bold')
    ax.set_ylabel(r"Mean $\sum u_i^2$ (Steady State)")
    ax.set_xlabel("Number of Agents ($N$)")
    
    # Vertical line for training reference
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training $N=30$")
    
    # Set both axes to log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Custom grid style from the reference script
    ax.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Ensure legend is visible
    ax.legend(title="Model", frameon=True, framealpha=0.9)

    save_path = OUTPUT_DIR / "effort_scaling_log_steady.pdf"
    plt.savefig(save_path) # rcParams handles dpi and bbox
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    solver_ts = Tesseract.from_image("solver_fkpp1d_decentralized:latest")
    
    print("Starting Effort Scaling Analysis (Steady State + Log Scale)...")
    df_results = evaluate_effort_scaling(solver_ts)
    
    # Save CSV for reference
    df_results.to_csv(OUTPUT_DIR / "effort_data_steady.csv", index=False)
    
    plot_effort_metrics(df_results)
    print("Analysis complete.")