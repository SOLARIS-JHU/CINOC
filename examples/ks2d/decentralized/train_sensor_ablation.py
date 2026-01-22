"""
Phase 3: Sensor Dimension Ablation

Trains 6 models with different local patch sizes to understand the relationship
between receptive field size and zero-shot scalability.

Hypothesis: Smaller patches enable better scalability but may struggle with
sparse configurations; larger patches work well when sparse but may overfit.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from train_utils_ks2d import train_ks2d
from models.policy_ks2d import DecentralizedKS2DControlNet

# Patch Sizes to Test (in pixels)
PATCH_SIZES = [6, 8, 12, 16, 20, 24]  # 12 is baseline

# Receptive Field Coverage:
# - 6×6 pixels = 3.0 domain units (sees just own cell)
# - 12×12 pixels = 6.0 domain units (sees ~2 neighboring agents) - BASELINE
# - 24×24 pixels = 12.0 domain units (sees ~4 neighboring agents)

# Fixed Training Constants
TRAIN_CONSTANTS = {
    'N_grid': 64,
    'L_domain': 32.0,
    'dt': 0.005,
    'substeps': 20,
    'T_steps': 50,
    'n_agents': 100,  # 10×10 grid
    'batch_size': 4,
    'epochs': 500,
    'pool_size': 500,
    # Fixed low noise for clean comparison
    'noise_u': 0.05,
    'noise_z': 0.025,
}


def main():
    print("="*70)
    print("KS2D Sensor Dimension Ablation Study")
    print("="*70)
    print(f"\nTraining {len(PATCH_SIZES)} models with different patch sizes...")
    print(f"Patch sizes: {PATCH_SIZES}")
    print(f"Training config: {TRAIN_CONSTANTS}\n")

    # Output directory
    save_dir = Path("figures/sensor_ablation")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train each model
    for patch_size in PATCH_SIZES:
        print("\n" + "="*70)
        print(f"Training: Patch Size = {patch_size}px")
        print(f"  Receptive field: {patch_size * TRAIN_CONSTANTS['L_domain'] / TRAIN_CONSTANTS['N_grid']:.1f} domain units")
        print("="*70)

        # Create model with specific patch size
        model = DecentralizedKS2DControlNet(
            features=(64, 128),
            domain_size=(TRAIN_CONSTANTS['L_domain'], TRAIN_CONSTANTS['L_domain']),
            u_max=5.0,
            patch_size=patch_size  # VARIES
        )

        params, metrics = train_ks2d(
            **TRAIN_CONSTANTS,
            model=model,  # Pass custom model
            patch_size=patch_size,  # For tracking
            save_dir=str(save_dir),
            params_file=f"patch_{patch_size}_params.msgpack",
            plot_file=f"patch_{patch_size}_training.png",
            plot_metrics=True,
            verbose=True
        )

        print(f"✓ Patch {patch_size}px training complete")
        print(f"  Final tracking MSE: {metrics[-1, 1]:.6f}")
        print(f"  Final effort: {metrics[-1, 2]:.6f}")

    print("\n" + "="*70)
    print("All models trained successfully!")
    print(f"Models saved in: {save_dir}")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python visualize_sensor_ablation.py")
    print("  2. Examine scalability curves in figures/")


if __name__ == "__main__":
    main()
