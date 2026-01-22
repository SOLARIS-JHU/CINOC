"""
Quick end-to-end test of the training pipeline.
Tests with minimal epochs to verify everything works.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from train_utils_ks2d import train_ks2d

print("="*70)
print("Quick Training Pipeline Test")
print("="*70)
print("\nTesting with 10 epochs to verify end-to-end functionality...")

save_dir = Path("test_output")
save_dir.mkdir(exist_ok=True)

try:
    params, metrics = train_ks2d(
        N_grid=64,
        L_domain=32.0,
        dt=0.005,
        substeps=20,
        T_steps=50,
        n_agents=100,
        batch_size=2,  # Reduced for speed
        epochs=10,     # Very few epochs
        pool_size=100, # Reduced pool
        noise_u=0.05,
        noise_z=0.025,
        save_dir=str(save_dir),
        params_file="test_params.msgpack",
        plot_file="test_training.png",
        plot_metrics=True,
        verbose=True
    )

    print("\n" + "="*70)
    print("✓ SUCCESS: Training pipeline works!")
    print("="*70)
    print(f"\nFinal metrics:")
    print(f"  Loss: {metrics[-1, 0]:.6f}")
    print(f"  Tracking MSE: {metrics[-1, 1]:.6f}")
    print(f"  Effort: {metrics[-1, 2]:.6f}")

    print(f"\nTest outputs saved in: {save_dir}")
    print("You can safely delete this directory.")

except Exception as e:
    print("\n" + "="*70)
    print("✗ FAILED: Training pipeline encountered an error")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
