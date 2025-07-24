#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "dr_bcg/helper.h"
#include "dr_bcg/device_buffer.h"

namespace dr_bcg
{
    void copy_upper_triangular(float *dst, float *src, const int m, const int n);

    std::tuple<std::vector<float>, int> dr_bcg(
        const std::vector<float> &A,
        const std::vector<float> &X,
        const std::vector<float> &B,
        const int m,
        const int n,
        const float tolerance,
        const int max_iterations);

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

    void get_xi(
        cusolverDnHandle_t &cusolverH, cusolverDnParams_t &cusolverParams, cublasHandle_t &cublasH,
        const int m, const int n, DeviceBuffer &d, const float *d_A);

    void get_sigma(cublasHandle_t cublasH, int n, DeviceBuffer &d);

    void get_s(cublasHandle_t cublasH, int m, int n, DeviceBuffer &d);

    void get_w_zeta(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &cusolverParams, cublasHandle_t &cublasH,
                    const int m, const int n, DeviceBuffer &d, const float *d_A);

    void residual(cublasHandle_t &cublasH, float *d_residual, const float *B, const int m, const float *d_A, const float *d_X);

    void get_next_X(cublasHandle_t &cublasH, const int m, const int n, const float *d_s, const float *d_xi, float *d_temp, const float *d_sigma, float *d_X);

    void quadratic_form(cublasHandle_t &cublasH, const int m, const int n, const float *d_s, const float *d_A, float *d_work, float *d_y);

    void thin_qr(
        cusolverDnHandle_t &cusolverH,
        cusolverDnParams_t &params,
        cublasHandle_t &cublasH,
        float *Q,
        float *R,
        const int m,
        const int n,
        const float *A);

    void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A);

    void get_R(cublasHandle_t &cublasH, float *h_R, const int n, const int m, const float *A, const float *X, const float *B);

    void invert_square_matrix(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *A, const int n);
}
