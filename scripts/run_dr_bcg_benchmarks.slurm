#!/bin/bash
#SBATCH --job-name="dr_bcg_benchmark"
#SBATCH --output="output/dr_bcg_benchmark.%j.%N.out"
#SBATCH --error="output/dr_bcg_benchmark.%j.%N.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 00:30:00

srun build/benchmark/dr_bcg_benchmark --benchmark_counters_tabular=true --benchmark_out_format=json --benchmark_out=dr_bcg.bench.json