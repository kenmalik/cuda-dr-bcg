#!/bin/bash
#SBATCH --job-name="dr_bcg_microbenchmark"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 08:00:00

if [[ -z ${BIN_DIR} ]]; then
    echo "cannot read matrices: BIN_DIR is not defined"
    exit 1
fi
if ! [[ -d ${BIN_DIR} ]]; then
    echo "cannot read matrices: BIN_DIR is not a directory"
    exit 1
fi

benchmarks=`ls ${BIN_DIR} | grep ".bin"`
if [ ${#benchmarks[@]} -eq 0 ]; then
    echo "matrices not found: BIN_DIR does not contain matrix .bin files"
    exit 1
fi

results_dir="${OUT_DIR}/microbenchmark_${SLURM_JOB_ID}_dr-bcg"
mkdir ${results_dir}
if [ $? -ne 0 ]; then
    echo "Could not create directory ${results_dir} to store results"
    exit 1
fi

for benchmark in ${benchmarks[@]}; do
    echo "Benchmarking ${benchmark}..."

    srun -o "${results_dir}/${benchmark%%.*}.json" \
        -e "${results_dir}/${benchmark%%.*}.err" \
        build/benchmark/dr_bcg_benchmark \
        --benchmark_filter="get_" -d "${BIN_DIR}/${benchmark}" --benchmark_format=json
    
    error_code=$?
    if [ ${error_code} -ne 0 ]; then
        echo "Error benchmarking against ${benchmark} with error code ${error_code}"
    else
        echo "Successfully benchmarked against ${benchmark}"
    fi
done