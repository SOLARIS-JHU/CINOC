# KS2D Ablation Studies

This directory contains three ablation studies for the 2D Kuramoto-Sivashinsky (KS2D) decentralized control system.

## Overview

All studies follow the established patterns from 1D examples (FKPP, KS1D) and test different aspects of the decentralized control approach:

1. **Noise Robustness (Decoupled)** - Separates actuator vs. state/sensor noise effects
2. **Sensor Dimension Ablation** - Varies local patch size for decentralized sensing
3. **Agent Count Transfer** - Tests zero-shot scalability across agent counts

## Quick Start

### Run All Studies

```bash
./run_all_ablations.sh
```

### Run Individual Studies

```bash
# Phase 1: Verify noise injection works
python test_noise_injection.py

# Phase 2: Noise robustness
python train_noise_decoupled.py
python visualize_noise_decoupled.py

# Phase 3: Sensor ablation
python train_sensor_ablation.py
python visualize_sensor_ablation.py

# Phase 4: Agent transfer
python train_transfer.py
python visualize_transfer.py
```

## Phase 1: Foundation

**Files:**
- `test_noise_injection.py` - Verification test

**Modifications:**
- `tesseracts/ks2d/solver.py` - Added noise injection to solver
- `dynamics_dual.py` - Pass noise parameters through wrapper
- `train_utils_ks2d.py` - Modular training utilities

**Key Changes:**
The solver now supports:
- `key`: JAX PRNG key for deterministic noise
- `noise_u`: Actuator noise magnitude (added to control signal)
- `noise_z`: Sensor noise magnitude (added to state observation)

Noise is injected at the appropriate points:
1. Sensor noise: Added to state BEFORE policy sees it
2. Actuator noise: Added to control BEFORE physics applies it

## Phase 2: Noise Robustness Study

**Files:**
- `train_noise_decoupled.py` - Trains 7 specialist models
- `visualize_noise_decoupled.py` - Evaluates cross-robustness

**Models Trained (7 total):**

```python
"baseline_clean":  noise_u=0.0,  noise_z=0.0

# Actuator Specialists
"actuator_low":    noise_u=0.05, noise_z=0.0
"actuator_mid":    noise_u=0.2,  noise_z=0.0
"actuator_high":   noise_u=1.0,  noise_z=0.0

# State Specialists
"state_low":       noise_u=0.0,  noise_z=0.025
"state_mid":       noise_u=0.0,  noise_z=0.1
"state_high":      noise_u=0.0,  noise_z=0.5
```

**Test Scenarios (9 total):**
- 3 state-only scenarios (varying z, u=0)
- 3 actuator-only scenarios (varying u, z=0)
- 3 combined scenarios (both u and z)

**Evaluation:**
- Test agent counts: [60, 80, 100, 120, 140]
- 20 test samples per configuration
- Total: 7 models × 9 scenarios × 5 counts × 20 samples = 6,300 evaluations

**Outputs:**
```
figures/noise_experiments/decoupled_robustness/
├── baseline_clean_params.msgpack
├── actuator_low_params.msgpack
├── ... (7 models total)
├── noise_robustness_results.csv
├── scalability_State_Low.png
├── scalability_Actuator_High.png
├── ... (9 scalability curves)
└── cross_robustness_heatmap.png
```

**Key Insights:**
- Which noise source (actuator vs state) is more critical?
- Do specialist models generalize to other noise types?
- How does noise robustness affect scalability?

## Phase 3: Sensor Dimension Ablation

**Files:**
- `train_sensor_ablation.py` - Trains 6 models with different patch sizes
- `visualize_sensor_ablation.py` - Evaluates scalability

**Patch Sizes Tested:**

```python
PATCH_SIZES = [6, 8, 12, 16, 20, 24]  # pixels

# Receptive Field Coverage:
# 6×6   = 3.0 domain units (sees just own cell)
# 12×12 = 6.0 domain units (sees ~2 neighbors) - BASELINE
# 24×24 = 12.0 domain units (sees ~4 neighbors)
```

**Training Configuration:**
- Fixed agent count: 100 (10×10 grid)
- Fixed low noise: u=0.05, z=0.025
- Training epochs: 500

**Evaluation:**
- Test agent counts: [49, 64, 81, 100, 121, 144, 169] (7²...13²)
- 50 test samples per configuration
- Total: 6 models × 7 counts × 50 samples = 2,100 evaluations

**Outputs:**
```
figures/sensor_ablation/
├── patch_6_params.msgpack
├── ... (6 models)
├── sensor_ablation_results.csv
├── sensor_scalability_curve.png
├── sensor_relative_performance.png
└── sensor_spacing_analysis.png
```

**Key Insights:**
- Optimal patch size for zero-shot scalability
- Trade-off between local vs global information
- How receptive field relates to agent spacing

## Phase 4: Agent Count Transfer

**Files:**
- `train_transfer.py` - Trains single baseline model
- `visualize_transfer.py` - Evaluates zero-shot transfer

**Training:**
- Single model trained with N=100 agents (10×10 grid)
- Clean training (no noise)
- 500 epochs

**Evaluation:**
- Test agent counts: [36, 49, 64, 81, 100, 121, 144, 169, 196] (6²...14²)
- 20 test samples per configuration
- Total: 9 counts × 20 samples = 180 evaluations

**Outputs:**
```
figures/agent_transfer/
├── baseline_n100_params.msgpack
├── transfer_results.csv
├── transfer_scalability.png
├── transfer_efficiency.png
├── transfer_trajectories.png
└── transfer_energy_decay.png
```

**Metrics:**
- Final tracking MSE vs agent count
- Energy decay rate
- Efficiency (MSE per agent)
- Visual trajectory comparison

**Key Insights:**
- How well does the policy transfer to different densities?
- Is performance degradation acceptable (< 10× baseline)?
- Optimal agent spacing for control authority

## Training Parameters

All studies use consistent physics parameters:

```python
N_grid = 64          # Spatial resolution
L_domain = 32.0      # Domain size
dt = 0.005           # Physics timestep
substeps = 20        # Physics steps per control step
T_steps = 50         # Control steps per trajectory
sigma = 1.2          # Actuator width

# Training
batch_size = 4
epochs = 500
pool_size = 500      # Initial condition pool
```

**Computational Cost:**
- Single training run: ~500 epochs × 5s/epoch = ~40 minutes
- Phase 2: 7 models = ~5 hours training + 2 hours evaluation
- Phase 3: 6 models = ~4 hours training + 1 hour evaluation
- Phase 4: 1 model = ~40 minutes training + 30 minutes evaluation
- **Total: ~14 hours**

## Noise Scaling Rationale

**2D KS States:** Typically range ~[-3, 3] during chaos

**Actuator Noise Levels:**
- Low (0.05): 1% of u_max=5.0
- Mid (0.2): 4% of u_max
- High (1.0): 20% of u_max

**State Noise Levels:**
- Low (0.025): 0.8% of typical range [-3, 3]
- Mid (0.1): 3.3% of range
- High (0.5): 16.7% of range

These are chosen to be challenging but not overwhelming for the 2D system.

## File Structure

```
examples/ks2d/decentralized/
├── train.py                          # Original training script
├── visualize.py                      # Original visualization
├── data_utils.py                     # Data generation
├── dynamics_dual.py                  # MODIFIED: noise support
│
├── train_utils_ks2d.py               # NEW: Modular training
├── test_noise_injection.py           # NEW: Verification
│
├── train_noise_decoupled.py          # NEW: Phase 2 training
├── visualize_noise_decoupled.py      # NEW: Phase 2 evaluation
│
├── train_sensor_ablation.py          # NEW: Phase 3 training
├── visualize_sensor_ablation.py      # NEW: Phase 3 evaluation
│
├── train_transfer.py                 # NEW: Phase 4 training
├── visualize_transfer.py             # NEW: Phase 4 evaluation
│
├── run_all_ablations.sh              # NEW: Master script
├── ABLATION_README.md                # This file
│
└── figures/
    ├── noise_experiments/
    │   └── decoupled_robustness/
    ├── sensor_ablation/
    └── agent_transfer/
```

## Success Metrics

### Noise Robustness
- ✓ Actuator specialists outperform baseline in pure actuator noise
- ✓ State specialists outperform baseline in pure state noise
- ✓ 9×7 heatmap identifies critical noise sources
- ✓ Scalability curves show robustness vs agent count

### Sensor Ablation
- ✓ Identify optimal patch size (~2× agent spacing)
- ✓ Demonstrate trade-off: small patches → better scalability
- ✓ Show performance degradation outside training density

### Agent Transfer
- ✓ Policy maintains MSE < 10× baseline when agents vary ±50%
- ✓ Energy decay demonstrates robustness across densities
- ✓ Visual comparison shows qualitative control quality

## Dependencies

All scripts use:
- JAX (with x64 precision enabled)
- Flax (for model serialization)
- Optax (for optimization)
- Matplotlib (for plotting)
- Pandas (for metrics)
- tqdm (for progress bars)

## Tips for Running

**Quick Testing:**
Reduce epochs to 50 for exploratory runs:
```python
# In each train_*.py file, modify:
TRAIN_CONSTANTS['epochs'] = 50
```

**Memory Issues:**
If you encounter OOM errors:
- Reduce `batch_size` from 4 to 2
- Reduce `N_TEST_SAMPLES` in visualization scripts

**Parallel Execution:**
Phases 2-4 are independent after Phase 1. You can run them in parallel:
```bash
python train_noise_decoupled.py &
python train_sensor_ablation.py &
python train_transfer.py &
wait
```

## Troubleshooting

**"Model not found" error:**
- Make sure training script completed successfully
- Check that params files exist in figures/ subdirectories

**NaN in training:**
- Check learning rate schedule (may need to reduce peak)
- Verify noise levels aren't too high
- Check gradient clipping is enabled

**Slow evaluation:**
- Reduce `N_TEST_SAMPLES` in visualization scripts
- Use fewer test agent counts
- Evaluation is not JIT-compiled, so first run is slow

## References

This implementation follows patterns from:
- `examples/ks1d/decentralized/train_utils.py` - Modular training
- `examples/fkpp1d/centralized/visualize_robustness_transfer.py` - Evaluation patterns
- `tesseracts/ks1d/solver.py` - Noise injection mechanism

## Contact

For questions or issues, see the main project README or CLAUDE.md.
