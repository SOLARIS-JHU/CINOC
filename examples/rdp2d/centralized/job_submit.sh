#!/bin/bash -l
#SBATCH --job-name=RDPC 
#SBATCH -t 2:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH -A sgoswam4_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=droysar1@jhu.edu

source $HOME/jax_torch2.venv/bin/activate

cd /home/droysar1/scr4_sgoswam4/Dibakar/multi_agent_dpc/Multi-Agent-DPC/examples/rdp2d/centralized
python3 -u train.py