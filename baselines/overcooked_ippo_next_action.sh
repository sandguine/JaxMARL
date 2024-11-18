#!/bin/bash
#SBATCH --job-name=sr-ser
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=50gb
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=default

export XLA_PYTHON_CLIENT_MEM_FRACTION=.9
python baselines/IPPO/ippo_cnn_overcooked_action_aware_wrapper.py