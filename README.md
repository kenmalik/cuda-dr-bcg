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

## Running Examples

This library was initially developed on a system running SLURM and hence contains some conveniences for running examples.

The Makefile in the root directory contains a target `run` which will run a SLURM script using `srun`.
The Makefile requires you to define an environment variable which defines the account you will run the script on.

Run the example like so, replacing `my-acct-num` with your account identifier:

```bash
export ACCOUNT=my-acct-num
make run
```
