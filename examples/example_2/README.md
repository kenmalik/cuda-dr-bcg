# Example 2

This example runs DR-BCG using the sparse matrix interface. The `n` by `n` matrix `A` is read from a `.mat` file containing a matrix from the SuiteSparse collection. The initial `X` guess is set to an `n` by `s` matrix of zeros. `B` is set to an `n` by `s` matrix of ones.

## Running

### Direct Invocation

Assuming you are in the `build/` directory, you can run the example with the following arguments.

```bash
examples/example_2/example_2 [path_to_mat_file] [OPTIONAL block_size]
```

If not provided, the block size defaults to `1`.

### As a SLURM Job

In the `scripts/` directory, there exists a SLURM script `profile_example_2.slurm` which allows you to profile this example against a directory containing SuiteSparse `.mat` files and a range of block sizes (`s`).

To use it, define the following environment variables:

```bash
export OUT_DIR=path/to/output/directory
export MAT_DIR=path/to/suitesparse/directory
```

Replace `path/to/output/directory` with the directory you would like `nsys` to output raw data and CSV files containing profile data.

Replace `path/to/suitesparse/directory` with the directory containing the SuiteSparse `.mat` files you would like to profile against.

After defining these variables, run the script like so:

```bash
sbatch --account=your-account-id scripts/profile_example_2.slurm
```