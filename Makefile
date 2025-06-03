.PHONY: build
build:
	cmake --build build

run: build
	sbatch --account=${ACCOUNT} run_dr-bcg.slurm
	squeue -u ${USER}