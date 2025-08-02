.PHONY: build
build:
	cmake --build build

run: build
	sbatch --account=${ACCOUNT} -o ${OUT_DIR}/%x.%j.%N.out -e ${OUT_DIR}/%x.%j.%N.err scripts/run_dr_bcg.slurm
	squeue -u ${USER}

.PHONY: benchmark
benchmark: build
	sbatch --account=${ACCOUNT} -o ${OUT_DIR}/%x.%j.%N.out -e ${OUT_DIR}/%x.%j.%N.err scripts/run_dr_bcg_benchmarks.slurm
	squeue -u ${USER}

suitesparse: build
	if [[ -n "${SUITESPARSE_DIR}" ]]; then \
		sbatch --account=${ACCOUNT} -o ${OUT_DIR}/%x.%j.%N.out -e ${OUT_DIR}/%x.%j.%N.err scripts/check_against_suitesparse.slurm; \
		squeue -u ${USER}; \
	else \
		echo "ERROR: Variable SUITESPARSE_DIR needs to be defined"; \
	fi
