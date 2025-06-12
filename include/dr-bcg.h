#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

int dr_bcg(float *A, const int n, const float *x, const float *b, const float tolerance, const int max_iterations);

void qr_decomposition(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *q, float *r, const int n, float *A, const float *b);

void get_r(cublasHandle_t &cublasH, float *h_r, const int &n, const float *A, const float *x, const float *b);
