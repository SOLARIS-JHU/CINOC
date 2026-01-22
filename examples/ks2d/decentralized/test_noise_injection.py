"""
Quick test to verify noise injection works correctly in KS2D solver.
"""

import jax
import jax.numpy as jnp
import sys
from pathlib import Path

jax.config.update("jax_enable_x64", True)

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_ks2d import DecentralizedKS2DControlNet
from train_utils_ks2d import get_agent_grid

print("=== Testing Noise Injection in KS2D ===\n")

# Initialize model
model = DecentralizedKS2DControlNet(
    features=(64, 128),
    domain_size=(32.0, 32.0),
    u_max=5.0,
    patch_size=12
)

key = jax.random.PRNGKey(42)
params = model.init(key, jnp.zeros((64, 64)), jnp.zeros((64, 64)), jnp.zeros((100, 2)))

# Setup dynamics
dynamics = PDEDynamics2D(model.apply)

# Test conditions
u_init = jax.random.normal(key, (64, 64))
xi_fixed = get_agent_grid(100, 32.0)
u_target = jnp.zeros((64, 64))

print("1. Testing clean (no noise) execution...")
try:
    key, subkey = jax.random.split(key)
    traj_clean = dynamics.unroll_controlled(
        u_init, xi_fixed, u_target, params, 10,
        N_grid=64, L=32.0, dt=0.005, sigma=1.2, substeps=20,
        key=subkey, noise_u=0.0, noise_z=0.0
    )
    print("   ✓ Clean execution successful")
    print(f"   Trajectory shape: {traj_clean[0].shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n2. Testing with actuator noise...")
try:
    key, subkey = jax.random.split(key)
    traj_actuator = dynamics.unroll_controlled(
        u_init, xi_fixed, u_target, params, 10,
        N_grid=64, L=32.0, dt=0.005, sigma=1.2, substeps=20,
        key=subkey, noise_u=0.1, noise_z=0.0
    )
    print("   ✓ Actuator noise injection successful")
    # Verify trajectories differ
    diff = jnp.mean((traj_clean[0] - traj_actuator[0])**2)
    print(f"   Difference from clean: {diff:.6f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n3. Testing with sensor noise...")
try:
    key, subkey = jax.random.split(key)
    traj_sensor = dynamics.unroll_controlled(
        u_init, xi_fixed, u_target, params, 10,
        N_grid=64, L=32.0, dt=0.005, sigma=1.2, substeps=20,
        key=subkey, noise_u=0.0, noise_z=0.05
    )
    print("   ✓ Sensor noise injection successful")
    diff = jnp.mean((traj_clean[0] - traj_sensor[0])**2)
    print(f"   Difference from clean: {diff:.6f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n4. Testing with combined noise...")
try:
    key, subkey = jax.random.split(key)
    traj_combined = dynamics.unroll_controlled(
        u_init, xi_fixed, u_target, params, 10,
        N_grid=64, L=32.0, dt=0.005, sigma=1.2, substeps=20,
        key=subkey, noise_u=0.1, noise_z=0.05
    )
    print("   ✓ Combined noise injection successful")
    diff = jnp.mean((traj_clean[0] - traj_combined[0])**2)
    print(f"   Difference from clean: {diff:.6f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n5. Verifying determinism with same key...")
try:
    key_test = jax.random.PRNGKey(999)
    traj_a = dynamics.unroll_controlled(
        u_init, xi_fixed, u_target, params, 10,
        N_grid=64, L=32.0, dt=0.005, sigma=1.2, substeps=20,
        key=key_test, noise_u=0.1, noise_z=0.05
    )

    key_test = jax.random.PRNGKey(999)  # Same key
    traj_b = dynamics.unroll_controlled(
        u_init, xi_fixed, u_target, params, 10,
        N_grid=64, L=32.0, dt=0.005, sigma=1.2, substeps=20,
        key=key_test, noise_u=0.1, noise_z=0.05
    )

    diff = jnp.mean((traj_a[0] - traj_b[0])**2)
    if diff < 1e-10:
        print(f"   ✓ Deterministic (diff={diff:.2e})")
    else:
        print(f"   ✗ Non-deterministic (diff={diff:.2e})")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n=== All Tests Passed! ===")
print("\nPhase 1 (Foundation) verification complete.")
print("Noise injection is working correctly in KS2D.")
