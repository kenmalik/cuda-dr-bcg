#pragma once

#include <optional>
#include <vector>
#include <string>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__);
#define CUSOLVER_CHECK(val) check((val), #val, __FILE__, __LINE__);
#define CUBLAS_CHECK(val) check((val), #val, __FILE__, __LINE__);
#define CUSPARSE_CHECK(val) check((val), #val, __FILE__, __LINE__);

void check(cudaError_t err, const char *const func, const char *const file, const int line);
void check(cusolverStatus_t err, const char *const func, const char *const file, const int line);
void check(cublasStatus_t err, const char *const func, const char *const file, const int line);
void check(cusparseStatus_t err, const char *const func, const char *const file, const int line);

void fill_random(float *mat, const int rows, const int cols, const std::optional<int> seed = std::nullopt);

void fill_spd(float *mat, const int n, const std::optional<int> seed = std::nullopt);

void print_matrix(const float *mat, const int rows, const int cols);

void print_device_matrix(const float *d_mat, const int rows, const int cols);

void print_sparse_matrix(const cusparseHandle_t &cusparseH, const cusparseSpMatDescr_t &sp_mat);

void check_nan(const float *d_arr, size_t size, std::string step);

std::vector<double> read_matrix_bin(std::string filename);

void copy_upper_triangular(float *dst, float *src, const int m, const int n);

void invert_square_matrix(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *A, const int n);

void thin_qr(
    cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &params,
    cublasHandle_t &cublasH,
    float *Q,
    float *R,
    const int m,
    const int n,
    const float *A);

void qr_factorization(
    cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &params,
    float *Q,
    float *R,
    const int m,
    const int n,
    const float *A);

void sptri_left_multiply(
    const cusparseHandle_t &cusparseH,
    cusparseDnMatDescr_t &C,
    cusparseOperation_t opA,
    const cusparseSpMatDescr_t &A,
    const cusparseDnMatDescr_t &B);