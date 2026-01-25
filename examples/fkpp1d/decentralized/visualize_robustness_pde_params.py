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

jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

# Output Directory
OUTPUT_DIR = Path("figures/pde_params")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Imports ---
from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

# --- Config ---
# Dense range of agents for the x-axis
TEST_AGENT_COUNTS = [20, 30, 40, 50, 60, 70, 80, 90, 100]
N_PDE = 100
T_STEPS = 300
N_TEST_SAMPLES = 50 

# Parameter Sets
NU_VALUES = [0., 0.001, 0.002, 0.003, 0.004, 0.005]
RHO_VALUES = [3, 4, 5, 6, 7, 8]

# Defaults
DEFAULT_NU = 0.005
DEFAULT_RHO = 3.0

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

def run_ablation(dynamics, params, z_init_batch, z_target_batch, nu_list, rho_list, varied_param_name):
    """
    Runs the ablation loop using PDEDynamics wrapper.
    """
    results = []
    
    # Determine which list to iterate
    iter_list = nu_list if varied_param_name == 'nu' else rho_list
    
    print(f"--- Running Ablation for {varied_param_name} ---")
    
    for val in iter_list:
        # Set parameters: one varies, the other stays default
        current_nu = val if varied_param_name == 'nu' else DEFAULT_NU
        current_rho = val if varied_param_name == 'rho' else DEFAULT_RHO
        
        for n_agents in TEST_AGENT_COUNTS:
            print(f"  Param: {val} | Agents: {n_agents}")
            
            # Create agent positions
            xi_test = jnp.linspace(0.1, 0.9, n_agents)
            xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))

            # Define JIT-compiled batch runner
            @jax.jit
            def run_batch(z_i, z_t, xi_i):
                def single_run(zi, zt, xii):
                    # Call the wrapper with explicit physics params
                    z_traj, _, _, _ = dynamics.unroll_controlled(
                        zi, xii, zt, params, T_STEPS, 
                        key=jax.random.PRNGKey(0),
                        noise_u=0.0, 
                        noise_z=0.0,
                        nu=current_nu,  # Inject dynamic nu
                        rho=current_rho # Inject dynamic rho
                    )
                    return jnp.mean((z_traj[-1] - zt)**2)
                return jax.vmap(single_run)(z_i, z_t, xi_i)

            mses = run_batch(z_init_batch, z_target_batch, xi_batch)
            
            results.append({
                "Agents": n_agents,
                "MSE": float(jnp.mean(mses)),
                "Value": str(val), # String for categorical legend
                "Parameter": varied_param_name
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    # 1. Initialize Dynamics Wrapper
    solver_ts = Tesseract.from_image("solver_fkpp1d_decentralized:latest")
    model = DecentralizedControlNet(features=(64, 64))
    dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False)
    
    # 2. Load "low_noise" Params
    param_path = Path("figures/noise_experiments/robustness_transfer/low_noise_params.msgpack")
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find low_noise params at {param_path}")
    
    params = load_params(model, param_path)
    
    # 3. Generate Test Data
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    _, z_init_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.2))(jax.random.split(k1, N_TEST_SAMPLES))
    _, z_target_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.4))(jax.random.split(k2, N_TEST_SAMPLES))

    # 4. Run Experiments
    # Experiment A: Vary Nu (Rho fixed at 3.0)
    df_nu = run_ablation(dynamics, params, z_init_test, z_target_test, NU_VALUES, [], 'nu')
    
    # Experiment B: Vary Rho (Nu fixed at 0.005)
    df_rho = run_ablation(dynamics, params, z_init_test, z_target_test, [], RHO_VALUES, 'rho')

    # 5. Plotting
    setup_paper_style()
    
    # Use 7.0 width for a full-width figure in a two-column paper 
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    # Plot Nu Ablation
    sns.lineplot(
        ax=axes[0], data=df_nu, x="Agents", y="MSE", 
        hue="Value", style="Value", markers=True, markersize=6, linewidth=1.5, palette="viridis"
    )
    axes[0].set_title(r"Sensitivity to Diffusion ($\nu$)", pad=10)
    axes[0].set_ylabel("Final Tracking Error (MSE)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Customize Legend
    # We remove the title and frame to make it cleaner, or keep it minimal
    axes[0].legend(title=r"$\nu$ Value", loc='best', framealpha=0.9, fontsize=9)

    # Plot Rho Ablation
    sns.lineplot(
        ax=axes[1], data=df_rho, x="Agents", y="MSE", 
        hue="Value", style="Value", markers=True, markersize=6, linewidth=1.5, palette="magma"
    )
    axes[1].set_title(r"Sensitivity to Growth Rate ($\rho$)", pad=10)
    axes[1].set_ylabel("") # Shared Y
    axes[1].set_yscale("log")
    axes[1].grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    axes[1].legend(title=r"$\rho$ Value", loc='best', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "physics_ablation_lownoise.pdf"
    plt.savefig(save_path)
    print(f"\nSaved combined ablation plot to {save_path}")