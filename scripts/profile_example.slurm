#!/bin/bash
#SBATCH --job-name="sparse_example_profile"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 48:00:00

mat_dir=$1
if ! [[ -d "$mat_dir" ]]; then
    echo "Could not find .mat directory: ${1}" >&2
    exit 1
fi

command="${@:2}"
if [[ ${#command[@]} == 0 ]]; then
    echo "No commands provided" >&2
    exit 1
fi

block_sizes=(1 $(seq 2 2 20))

readarray -d "" -t mats < <(find ${mat_dir} -maxdepth 1 -name "*.mat" -print0)

for mat_file in ${mats[@]}; do
    mat_name=$(basename ${mat_file} .mat)
    echo "Now profiling against ${mat_name}" >&2
    mat_formatted=$(echo "${command}" | sed "s#{MAT}#${mat_file}#g")
    
    echo "Mat formatted: ${mat_formatted}" >&2

    parent_dir="${SLURM_JOB_NAME}_${SLURM_JOB_ID}/${mat_name}"
    mkdir -p "${parent_dir}" || exit 1

    for s in ${block_sizes[@]}; do
        block_formatted=$(echo "${mat_formatted}" | sed "s#{BLOCK}#${s}#g")
        echo "Block formatted: ${block_formatted}" >&2

        output_base="${parent_dir}/${s}"

        echo "Profiling..."
        srun -t 01:00:00 -o "${output_base}.profile.out" -e "${output_base}.profile.err" \
            nsys profile -t nvtx -o "${output_base}" ${block_formatted}

        code=$?
        if [[ ${code} != 0 ]]; then
            echo "Profiling failed with code ${code} so stats not generated. Continuing..."
            continue
        fi
        echo "Profiling finished!" >&2

        read -r n < <(head -1 "${output_base}.profile.out" | cut -f 2 -d ' ' | tr '\n' ' ')

        echo "Generating report..."
        srun -t 00:05:00 -o "${output_base}.stats.out" -e "${output_base}.stats.err" \
            nsys stats --report nvtxsum --format csv --output "${output_base}_${n}" "${output_base}.nsys-rep"
        echo "Finished generating report!"
    done
done