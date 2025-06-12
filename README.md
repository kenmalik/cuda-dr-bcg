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

## Running Examples

The Makefile in the root directory contains a target `run` which will run a SLURM script using `srun`.
The Makefile requires you to define an environment variable which defines the account you will run the script on.

Run the example like so, replacing `my-acct-num` with your account identifier:

```bash
export ACCOUNT=my-acct-num
make run
```
