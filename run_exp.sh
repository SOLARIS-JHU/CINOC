#!/bin/bash

# Define the list of folders
FOLDERS=(
    "/home/zanot/projects/Multi-Agent-DPC/examples/fkpp1d/decentralized/bench/"
    "/home/zanot/projects/Multi-Agent-DPC/examples/heat1d/decentralized/bench"
    "/home/zanot/projects/Multi-Agent-DPC/examples/heat2D/decentralized/bench"
    "/home/zanot/projects/Multi-Agent-DPC/examples/heat2D_obstacles/decentralized/bench"
    "/home/zanot/projects/Multi-Agent-DPC/examples/ks1d/decentralized/bench"
    # "/home/zanot/projects/Multi-Agent-DPC/examples/ks2d/decentralized/bench"
)

for FOLDER in "${FOLDERS[@]}"; do
    echo "========================================================"
    echo "Processing folder: $FOLDER"
    
    # Switch to the directory. If the directory doesn't exist, skip to the next one.
    cd "$FOLDER" || { echo "⚠️ Directory not found! Skipping to the next folder..."; continue; }

    # Run train_rl.py. If it crashes, print a warning but continue.
    echo ">>> Running train_rl.py..."
    python train_rl.py || echo "⚠️ ERROR: train_rl.py failed in $FOLDER. Continuing to the next script..."

    # Run train_marl.py. If it crashes, print a warning but continue.
    echo ">>> Running train_marl.py..."
    python train_marl.py || echo "⚠️ ERROR: train_marl.py failed in $FOLDER. Continuing to the next folder..."

done

echo "========================================================"
echo "Queue finished! All available scripts have been executed."