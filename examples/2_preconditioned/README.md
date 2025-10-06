# Example 2: Preconditioned DR-BCG

This example runs the PDR-BCG algorithm with `n` by `n` preconditioner `L`.

## Running

Assuming you are in the `build/` directory, you can run the example with the following arguments.

```bash
examples/2_preconditioned/preconditioned [path_to_mat_file] [path_to_preconditioner] [OPTIONAL block_size]
```

If not provided, the block size defaults to `1`.

## Matrix File Format

The mat file for sparse matrix `A` is assumed to come from the SuiteSparse Matrix
Collection. Such mat files generally have the structure `Problem>A`, meaning `A`
is nested in a MATLAB struct called `Problem`. The mat file for `L`, on the other
hand, is expected to have `L` stored at root level (not nested in any structs).