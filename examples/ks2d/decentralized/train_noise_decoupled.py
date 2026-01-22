"""
Phase 2: Noise Robustness Study (Decoupled)

Trains 7 specialist models to decouple actuator vs. state/sensor noise effects:
- 1 baseline (clean)
- 3 actuator specialists (vary noise_u, fix noise_z=0)
- 3 state specialists (vary noise_z, fix noise_u=0)

Each model is trained with specific noise configuration and evaluated on
all 9 test scenarios (cross-robustness analysis).
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from train_utils_ks2d import train_ks2d

# Noise Configurations (7 models)
NOISE_CONFIGS = {
    "baseline_clean": {"noise_u": 0.0, "noise_z": 0.0},

    # Actuator Specialists (vary u, fix z=0)
    "actuator_low":  {"noise_u": 0.05, "noise_z": 0.0},
    "actuator_mid":  {"noise_u": 0.2,  "noise_z": 0.0},
    "actuator_high": {"noise_u": 1.0,  "noise_z": 0.0},

    # State Specialists (vary z, fix u=0)
    "state_low":  {"noise_u": 0.0, "noise_z": 0.025},
    "state_mid":  {"noise_u": 0.0, "noise_z": 0.1},
    "state_high": {"noise_u": 0.0, "noise_z": 0.5},
}

# Fixed Training Constants
TRAIN_CONSTANTS = {
    'N_grid': 64,
    'L_domain': 32.0,
    'dt': 0.005,
    'substeps': 20,
    'T_steps': 50,
    'n_agents': 100,
    'batch_size': 4,
    'epochs': 500,
    'pool_size': 500,
}


def main():
    print("="*70)
    print("KS2D Noise Robustness Study (Decoupled)")
    print("="*70)
    print(f"\nTraining {len(NOISE_CONFIGS)} specialist models...")
    print(f"Training config: {TRAIN_CONSTANTS}\n")

    # Output directory
    save_dir = Path("figures/noise_experiments/decoupled_robustness")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train each model
    for model_name, noise_config in NOISE_CONFIGS.items():
        print("\n" + "="*70)
        print(f"Training: {model_name}")
        print(f"  Noise config: {noise_config}")
        print("="*70)

        params, metrics = train_ks2d(
            **TRAIN_CONSTANTS,
            **noise_config,
            save_dir=str(save_dir),
            params_file=f"{model_name}_params.msgpack",
            plot_file=f"{model_name}_training.png",
            plot_metrics=True,
            verbose=True
        )

        print(f"✓ {model_name} training complete")
        print(f"  Final tracking MSE: {metrics[-1, 1]:.6f}")
        print(f"  Final effort: {metrics[-1, 2]:.6f}")

    print("\n" + "="*70)
    print("All models trained successfully!")
    print(f"Models saved in: {save_dir}")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python visualize_noise_decoupled.py")
    print("  2. Examine cross-robustness matrices in figures/")


if __name__ == "__main__":
    main()
