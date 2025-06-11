#!/bin/bash
#SBATCH --job-name="dr-bcg"
#SBATCH --output="output/dr-bcg.%j.%N.out"
#SBATCH --error="output/dr-bcg.%j.%N.err"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 01:00:00

srun build/dr-bcg