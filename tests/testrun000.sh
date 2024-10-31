#!/bin/bash
#SBATCH --job-name=overcooked
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=50gb
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=default
source .jaxmarl_3106_test/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9
srun python test_jax_gpu.py