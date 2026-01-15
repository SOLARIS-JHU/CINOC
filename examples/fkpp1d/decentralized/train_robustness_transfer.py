"""
Robustness Transfer Experiment - Runner
Trains three variants of the ControlNet (Clean, Low Noise, High Noise)
using the shared train_utils module.
"""
from pathlib import Path
from train_utils import train

# Experiment Output Directory
EXPERIMENT_DIR = Path("figures/noise_experiments/robustness_transfer")

# --- Experiment Constants ---
# These match the original script
EXPERIMENT_CONSTANTS = {
    "n_pde": 100,
    "n_agents": 30,
    "batch_size": 32,
    "T_steps": 300,
    "epochs": 500,
    "R_safe": 0.05
}

# Define the three noise configurations
NOISE_CONFIGS = {
    "baseline":   {"noise_u": 0.0,  "noise_z": 0.0},
    "low_noise":  {"noise_u": 0.02, "noise_z": 0.01},
    "medium_noise": {"noise_u": 0.1,  "noise_z": 0.05},
    "high_noise": {"noise_u": 0.5,  "noise_z": 0.25},
}

def run_all():
    print(f"Starting Robustness Transfer Experiments...")
    print(f"Saving results to: {EXPERIMENT_DIR.resolve()}\n")

    for config_name, noise_vals in NOISE_CONFIGS.items():
        print(f"=== Running Configuration: {config_name} ===")
        
        # Run the training
        train(
            # Pass experiment constants
            **EXPERIMENT_CONSTANTS,
            
            # Pass noise specific to this run
            noise_u=noise_vals['noise_u'],
            noise_z=noise_vals['noise_z'],
            
            # Output settings
            save_repo=str(EXPERIMENT_DIR),
            net_params_filename=f"{config_name}_params",
            plot_filename=f"{config_name}_training_plot",
            plot_metrics=True
        )
        print(f"Completed {config_name}\n")

if __name__ == "__main__":
    run_all()