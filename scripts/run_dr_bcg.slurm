#!/bin/bash
#SBATCH --job-name="dr_bcg"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 00:30:00

srun build/examples/example_0/example_0