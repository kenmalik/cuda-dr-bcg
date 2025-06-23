#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace dr_bcg
{
    int dr_bcg(const float *A, const int m, const int n, float *X, const float *B, const float tolerance, const int max_iterations);

    void residual(cublasHandle_t &cublasH, float *d_residual, const float *B, const int m,  const float *d_A, const float *d_X);

    void next_X(cublasHandle_t &cublasH, const int m, const int n, const float *d_s, const float *d_xi, float *d_temp, const float *d_sigma, float *d_X);

    void quadratic_form(cublasHandle_t &cublasH, const int m, const int n, const float *d_s, const float *d_A, float *d_work, float *d_y);

    void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A);

    void get_R(cublasHandle_t &cublasH, float *h_R, const int n, const int m, const float *A, const float *X, const float *B);

    void invert_spd(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *A, const int64_t n);
}
