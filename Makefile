.PHONY: build
build:
	cmake --build build

run: build
	sbatch --account=${ACCOUNT} scripts/run_dr-bcg.slurm
	squeue -u ${USER}