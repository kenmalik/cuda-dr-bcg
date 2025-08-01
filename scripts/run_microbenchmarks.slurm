#!/bin/bash
#SBATCH --job-name="dr_bcg_microbenchmark"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 00:30:00

benchmarks=("get_xi" "get_next_X" "get_w_zeta" "get_s" "get_sigma")

for benchmark in ${benchmarks[@]}; do
    echo "Benchmarking ${benchmark}..."
    srun -o "${OUT_DIR}/${benchmark}.${SLURM_JOB_ID}.${SLURMD_NODENAME}.json" \
        -e "${OUT_DIR}/${benchmark}.${SLURM_JOB_ID}.${SLURMD_NODENAME}.err" \
        build/benchmark/dr_bcg_benchmark \
        --benchmark_filter="/${benchmark}/" --benchmark_format=json
    echo "Finished benchmarking ${benchmark}"
done