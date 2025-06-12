#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace dr_bcg
{
    int dr_bcg( float *A, const int n, const int m, const float *X, const float *B, const float tolerance, const int max_iterations);

    void qr_decomposition(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *q, float *r, const int n, float *A);

    void get_R(cublasHandle_t &cublasH, float *h_R, const int n, const int m, const float *A, const float *X, const float *B);
}
