#!/bin/bash
#SBATCH --job-name="suitesparse_check"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 5:00:00

source scripts/run_mat/.venv/bin/activate
srun python scripts/run_mat/test_runner.py ${SUITESPARSE_DIR} build/test/iterations_check {file} {m}
