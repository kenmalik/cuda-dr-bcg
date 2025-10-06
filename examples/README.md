# Examples

## Building

To build the examples, provide the `DR_BCG_BUILD_EXAMPLES` build flag when building using CMake. This can be done from the project root directory like so:

```bash
cmake -B build -S . -DDR_BCG_BUILD_EXAMPLES=ON
```

After building, this will add an additional `examples/` subdirectory under `build/` containing executables for each example.

## Running

See the `README.md` files in each example's directory for directions on how to run them.

### Profiling via a SLURM Job

In the `scripts/` directory, there exists a SLURM script `profile_example.slurm` which allows you to profile examples against a directory containing SuiteSparse `.mat` files and a range of block sizes (`s`).

The general format to using this script is like so:

```bash
sbatch --account=your-account-name scripts/profile_example.slurm /path/to/mat/directory [YOUR_COMMAND]
```

where `[YOUR_COMMAND]` is replaced by the command to call the example you wish to run (e.g. `/path/to/example/binary {MAT} {BLOCK}`).

Note that `{MAT}` and `{BLOCK}` are special arguments that will be replaced by `.mat` files in the specified mat directory and block sizes from the list of block sizes defined in the script. These arguments can be moved around and used multiple times in the command, if needed.