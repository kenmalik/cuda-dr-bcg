# CUDA DR-BCG

## Introduction

This is a CUDA implementation of the Dubrulle-R Block Conjugate Gradient (DR-BCG) algorithm for solving linear systems.

The implementation was originally derived from the following MATLAB code:

```matlab
function [X_final, iterations] = DR_BCG(A, B, X, tol, maxit)
    iterations = 0;
    R = B - A * X;
    [w, sigma] = qr(R,'econ');
    s = w;

    for k = 1:maxit
        iterations = iterations + 1;
        xi = (s' * A * s)^-1;
        X = X + s * xi * sigma;
        if (norm(B(:,1) - A * X(:,1)) / norm(B(:,1))) < tol
            break
        else
            [w, zeta] = qr(w - A * s * xi,'econ');
            s = w + s * zeta';
            sigma = zeta * sigma;
        end
    end
    X_final = X;
end
```

## Building

To simply build the library, run the following commands from the root directory:

```bash
cmake -B build -S .
cmake --build build
```

### Options

You can pass options when building the project for additional/altered functionality.

The following options adjust the behavior of the DR-BCG algorithm:

- `DR_BCG_USE_TENSOR_CORES`: Default `ON`. Enable TF32 calculation using Tensor Cores.
- `DR_BCG_USE_THIN_QR`: Default `OFF`. Use Thin QR procedure rather than cuSOLVER's standard QR procedure for the factorization portions of DR-BCG.

The following options build additional portions of the project. These are off by default:

- `DR_BCG_BUILD_BENCHMARKS`
- `DR_BCG_BUILD_EXAMPLES`
- `DR_BCG_BUILD_TESTS`

You can pass these when building the project. For example:

```bash
cmake -B build -S . -DDR_BCG_BUILD_EXAMPLES=ON -DDR_BCG_USE_TENSOR_CORES=ON
cmake --build build
```

## Running Examples

See [here](examples/README.md) for directions on building and running examples.

## Usage

The library currently provides two interfaces for the algorithm (the core solving portion is the same under the hood).

### From Host Memory

There exists a convenience wrapper for calling the algorithm using vectors:

```c++
std::tuple<std::vector<float>, int> dr_bcg(
    const std::vector<float> &A,
    const std::vector<float> &X,
    const std::vector<float> &B,
    const int m,
    const int n,
    const float tolerance,
    const int max_iterations);
```

You can call this interface like so:

```c++
auto [solution, iterations] = dr_bcg::dr_bcg(A, X, B, m, n, tolerance, max_iterations);
```

### From Device Memory

There also exists an interface for those who wish to call the routine using device memory pointers:

```c++
cusolverStatus_t dr_bcg(
    cusolverDnHandle_t cusolverH,
    cusolverDnParams_t cusolverParams,
    cublasHandle_t cublasH,
    int m,
    int n,
    const float *A,
    float *X,
    const float *B,
    float tolerance,
    int max_iterations,
    int *iterations);
```

See the convenience wrapper implementation for an example of how to call this interface.

## Building Tests and Benchmarks

By default, unit tests and benchmarks are not built alongside the main library.

To build them, build the project with the following flags:

```bash
cmake -B build -S . -DDR_BCG_BUILD_TESTS=ON -DDR_BCG_BUILD_BENCHMARKS=ON
```
