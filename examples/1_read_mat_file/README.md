# Example 1: Reading MAT Files

This example runs DR-BCG using the sparse matrix interface. The `n` by `n` matrix `A` is read from a `.mat` file containing a matrix from the SuiteSparse collection. The initial `X` guess is set to an `n` by `s` matrix of zeros. `B` is set to an `n` by `s` matrix of ones.

## Running

Assuming you are in the `build/` directory, you can run the example with the following arguments.

```bash
examples/1_read_mat_file/read_mat_file [path_to_mat_file] [OPTIONAL block_size]
```

If not provided, the block size defaults to `1`.