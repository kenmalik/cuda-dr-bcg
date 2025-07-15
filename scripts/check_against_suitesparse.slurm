#!/bin/bash
#SBATCH --job-name="suitesparse_check"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 00:30:00

source scripts/run_mat/.venv/bin/activate
srun python scripts/run_mat/test_runner.py -o ${OUT_DIR}/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.${SLURMD_NODENAME}.csv ${SUITESPARSE_DIR} build/examples/example_1/example_1 {file} {m}
