"""
Phase 4: Agent Count Transfer Study

Trains a single model with N=100 agents (10×10 grid) and evaluates zero-shot
transfer to different agent counts.

Tests the core hypothesis: decentralized policies with local sensing can
generalize across agent densities without retraining.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from train_utils_ks2d import train_ks2d

# Single Training Configuration
TRAIN_CONFIG = {
    'N_grid': 64,
    'L_domain': 32.0,
    'dt': 0.005,
    'substeps': 20,
    'T_steps': 50,
    'n_agents': 100,  # 10×10 grid
    'batch_size': 4,
    'epochs': 500,
    'pool_size': 500,
    # Clean training (no noise)
    'noise_u': 0.0,
    'noise_z': 0.0,
}


def main():
    print("="*70)
    print("KS2D Agent Count Transfer Study")
    print("="*70)
    print(f"\nTraining single model for zero-shot transfer...")
    print(f"Training config: {TRAIN_CONFIG}\n")

    # Output directory
    save_dir = Path("figures/agent_transfer")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"Training: Baseline (N={TRAIN_CONFIG['n_agents']} agents)")
    print("="*70)

    params, metrics = train_ks2d(
        **TRAIN_CONFIG,
        save_dir=str(save_dir),
        params_file="baseline_n100_params.msgpack",
        plot_file="baseline_n100_training.png",
        plot_metrics=True,
        verbose=True
    )

    print("\n" + "="*70)
    print("Training complete!")
    print(f"  Final tracking MSE: {metrics[-1, 1]:.6f}")
    print(f"  Final effort: {metrics[-1, 2]:.6f}")
    print(f"Model saved in: {save_dir}")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python visualize_transfer.py")
    print("  2. Examine scalability results in figures/")


if __name__ == "__main__":
    main()
