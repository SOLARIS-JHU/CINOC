#!/bin/bash
# Master script to run all KS2D ablation studies

set -e  # Exit on error

echo "=========================================="
echo "KS2D Ablation Studies - Master Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "train_noise_decoupled.py" ]; then
    echo "Error: Must run from examples/ks2d/decentralized/"
    exit 1
fi

# Parse arguments
QUICK_MODE=false
PHASE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--phase N]"
            echo "  --quick: Run with reduced epochs for testing"
            echo "  --phase N: Run only phase N (1-4)"
            exit 1
            ;;
    esac
done

# Modify scripts for quick mode
if [ "$QUICK_MODE" = true ]; then
    echo "Running in QUICK MODE (reduced epochs)"
    EPOCHS=50
    # This would require modifying the scripts to accept command-line args
    # For now, just warn the user
    echo "WARNING: Quick mode requires manual epoch reduction in scripts"
    echo "  Edit TRAIN_CONSTANTS['epochs'] = 50 in each train_*.py file"
    read -p "Press Enter to continue or Ctrl+C to abort..."
fi

# Phase 1: Foundation (already implemented, just verify)
if [ -z "$PHASE" ] || [ "$PHASE" = "1" ]; then
    echo "=========================================="
    echo "Phase 1: Foundation Verification"
    echo "=========================================="
    python test_noise_injection.py
    echo "✓ Phase 1 complete"
    echo ""
fi

# Phase 2: Noise Robustness Study
if [ -z "$PHASE" ] || [ "$PHASE" = "2" ]; then
    echo "=========================================="
    echo "Phase 2: Noise Robustness Study"
    echo "=========================================="

    echo "Training 7 models..."
    python train_noise_decoupled.py

    echo ""
    echo "Evaluating models..."
    python visualize_noise_decoupled.py

    echo "✓ Phase 2 complete"
    echo ""
fi

# Phase 3: Sensor Dimension Ablation
if [ -z "$PHASE" ] || [ "$PHASE" = "3" ]; then
    echo "=========================================="
    echo "Phase 3: Sensor Dimension Ablation"
    echo "=========================================="

    echo "Training 6 models with different patch sizes..."
    python train_sensor_ablation.py

    echo ""
    echo "Evaluating patch size sensitivity..."
    python visualize_sensor_ablation.py

    echo "✓ Phase 3 complete"
    echo ""
fi

# Phase 4: Agent Count Transfer
if [ -z "$PHASE" ] || [ "$PHASE" = "4" ]; then
    echo "=========================================="
    echo "Phase 4: Agent Count Transfer"
    echo "=========================================="

    echo "Training baseline model (N=100)..."
    python train_transfer.py

    echo ""
    echo "Evaluating zero-shot transfer..."
    python visualize_transfer.py

    echo "✓ Phase 4 complete"
    echo ""
fi

echo "=========================================="
echo "All Ablation Studies Complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - figures/noise_experiments/decoupled_robustness/"
echo "  - figures/sensor_ablation/"
echo "  - figures/agent_transfer/"
echo ""
echo "View results with:"
echo "  ls -R figures/"
