.PHONY: build
build:
	cmake --build build

run: build
	sbatch --account=${ACCOUNT} scripts/run_dr-bcg.slurm
	squeue -u ${USER}

.PHONY: benchmark
benchmark: build
	sbatch --account=${ACCOUNT} scripts/run_dr_bcg_benchmarks.slurm
	squeue -u ${USER}