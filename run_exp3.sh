#!/bin/bash

# Wait for 40 minutes
sleep 40m

# Change to the target directory. The '|| exit' ensures the script stops if the folder doesn't exist.
cd /home/zanot/projects/Multi-Agent-DPC/examples/density/centralized/ || exit

# Run the Python script
python train_new.py