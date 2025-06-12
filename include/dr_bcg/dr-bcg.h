#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace dr_bcg
{
    int dr_bcg( float *A, const int m, const int n, const float *X, const float *B, const float tolerance, const int max_iterations);

    void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A);

    void get_R(cublasHandle_t &cublasH, float *h_R, const int n, const int m, const float *A, const float *X, const float *B);
}
