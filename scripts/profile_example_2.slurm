#!/bin/bash
#SBATCH --job-name="sparse_example_profile"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 00:30:00

EXAMPLE_BINARY=build/examples/example_2/example_2

block_sizes=(1)
for i in $(seq 2 2 20); do
    block_sizes+=("${i}")
done
echo "Block sizes ${block_sizes[@]}"

if [[ -z "${OUT_DIR}" ]]; then
    echo "No output directory provided"
    exit 1
fi

if [[ -z "${MAT_DIR}" ]]; then
    echo "Mat file directory not provided"
    exit 1
fi

for file in $(ls ${MAT_DIR}); do
    mat_name=$(basename $file .mat)
    mat_path="${MAT_DIR}/${file}"
    echo "Now profiling against ${mat_name}"

    parent_dir="${OUT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/${mat_name}"
    mkdir -p "${parent_dir}" || exit 1

    for s in ${block_sizes[@]}; do
        output_base="${parent_dir}/${s}"

        echo "Profiling..."
        srun -t 00:01:00 -o "${output_base}.profile.out" -e "${output_base}.profile.err" \
            nsys profile -t nvtx -o "${output_base}" "${EXAMPLE_BINARY}" "${mat_path}"

        if [[ $? != 0 ]]; then
            echo "Profiling failed so stats not generated. Continuing..."
            continue
        fi
        echo "Profiling finished!"

        read -r n < <(head -1  "${output_base}.profile.out" | cut -f 2 -d ' ' | tr '\n' ' ')

        echo "Generating report..."
        srun -t 00:01:00 -o "${output_base}.stats.out" -e "${output_base}.stats.err" \
            nsys stats --report nvtxsum --format csv --output "${output_base}_${n}" "${output_base}.nsys-rep"
        echo "Finished generating report!"
    done
done