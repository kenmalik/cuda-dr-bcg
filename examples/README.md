# DR-BCG Examples

This directory contains examples demonstrating how to use the DR-BCG library.

## Simple Sparse Double Example

[simple_sparse_double.cu](simple_sparse_double.cu) demonstrates the basic usage of the non-preconditioned double-precision `dr_bcg` function for solving sparse linear systems.

### What the Example Does

The example:
1. Creates a 100x100 symmetric positive definite tridiagonal matrix (tridiag(-1, 2, -1))
2. Sets up a right-hand side vector of all ones
3. Solves the system Ax = b using DR-BCG with double precision
4. Prints convergence information and the solution

### Building the Examples

From the root directory of the project:

```bash
cmake -B build -S . -DDR_BCG_BUILD_EXAMPLES=ON
cmake --build build
```

### Running the Example

After building:

```bash
./build/examples/simple_sparse_double
```