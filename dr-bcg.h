#pragma once

void qr_decomposition(float *q, float *r, const int n, float *A, const float *b);

void get_r(cublasHandle_t &cublasH, float *h_r, const int &n, const float *A, const float *x, const float *b);
