#include <iostream>
#include <vector>
#include <tuple>
#include <string>

#include <nvtx3/nvtx3.hpp>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

/**
 * @brief Convenience wrapper for DR-BCG solver routine.
 *
 * Solves the block linear system AX = B using the DR-BCG algorithm, taking vectors and allocating device memory as required.
 *
 * @param A Host vector representing input matrix A (m x m)
 * @param X Host vector representing initial guess X (m x n)
 * @param B Host vector representing right-hand side B (m x n)
 * @param m m dimension
 * @param n n dimension
 * @param tolerance Relative residual tolerance for convergence
 * @param max_iterations Maximum number of iterations
 * @return Tuple containing the solution X (as a std::vector<float>) and the number of iterations performed
 */
std::tuple<std::vector<float>, int> dr_bcg::dr_bcg(
    const std::vector<float> &A,
    const std::vector<float> &X,
    const std::vector<float> &B,
    const int m,
    const int n,
    const float tolerance,
    const int max_iterations)
{
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t cusolverParams = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

    std::vector<float> X_final(m * n);
    int iterations = 0;

    float *d_A = nullptr;
    float *d_X = nullptr;
    float *d_B = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * m * n));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * m * m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(dr_bcg(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B, tolerance, max_iterations, &iterations));

    CUDA_CHECK(cudaMemcpy(X_final.data(), d_X, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverParams));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));

    return {X_final, iterations};
}

/**
 * @brief Main DR-BCG solver routine.
 *
 * Solves the block linear system AX = B using the DR-BCG algorithm on device pointers.
 *
 * @param cusolverH cuSOLVER handle
 * @param cusolverParams cuSOLVER params
 * @param cublasH cuBLAS handle
 * @param m m dimension
 * @param n n dimension
 * @param A Device pointer to input matrix A (m x m)
 * @param X Device pointer to initial guess X (m x n), overwritten with solution
 * @param B Device pointer to right-hand side B (m x n)
 * @param tolerance Relative residual tolerance for convergence
 * @param max_iterations Maximum number of iterations
 * @param iterations Pointer to int, overwritten with number of iterations performed
 * @return cuSOLVER status
 */
cusolverStatus_t dr_bcg::dr_bcg(
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
    int *iterations)
{
    NVTX3_FUNC_RANGE();

    DeviceBuffer d(m, n);

#ifdef USE_TENSOR_CORES
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TF32_TENSOR_OP_MATH));
#endif

    // We don't include d_R in device buffers because it is only used once at the beginning
    // of the algorithm.
    float *d_R;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(float) * m * n));

    // R = B - AX
    get_R(cublasH, d_R, m, n, A, X, B);

#ifdef USE_THIN_QR
    thin_qr(cusolverH, cusolverParams, cublasH, d.w, d.sigma, m, n, d_R);
#else
    qr_factorization(cusolverH, cusolverParams, d.w, d.sigma, m, n, d_R);
#endif

    CUDA_CHECK(cudaFree(d_R)); // Never used later

    // s = w
    CUDA_CHECK(cudaMemcpy(d.s, d.w, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

    float B1_norm;
    CUBLAS_CHECK(cublasSnrm2_v2(cublasH, m, B, 1, &B1_norm));

    *iterations = 0;
    while (*iterations < max_iterations)
    {
        nvtx3::scoped_range loop{"iteration"};

        (*iterations)++;

        // xi = (s' * A * s)^-1
        get_xi(cusolverH, cusolverParams, cublasH, m, n, d, A);

        // X = X + s * xi * sigma
        get_next_X(cublasH, m, n, d.s, d.xi, d.temp, d.sigma, X);

        // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
        float relative_residual_norm;
        residual(cublasH, d.residual, B, m, A, X);

        CUBLAS_CHECK(cublasSnrm2_v2(cublasH, m, d.residual, 1, &relative_residual_norm));
        relative_residual_norm /= B1_norm;

        if (relative_residual_norm < tolerance)
        {
            break;
        }
        else
        {
            nvtx3::scoped_range new_s_and_sigma{"get_new_s_and_sigma"};

            get_w_zeta(cusolverH, cusolverParams, cublasH, m, n, d, A);

            get_s(cublasH, m, n, d);

            get_sigma(cublasH, n, d);
        }
    }

    return CUSOLVER_STATUS_SUCCESS;
}

void dr_bcg::get_xi(
    cusolverDnHandle_t &cusolverH, cusolverDnParams_t &cusolverParams, cublasHandle_t &cublasH,
    const int m, const int n, DeviceBuffer &d, const float *d_A)
{
    NVTX3_FUNC_RANGE();

    quadratic_form(cublasH, m, n, d.s, d_A, d.temp, d.xi);
    invert_square_matrix(cusolverH, cusolverParams, d.xi, n);
}

void dr_bcg::get_sigma(cublasHandle_t cublasH, int n, DeviceBuffer &d)
{
    NVTX3_FUNC_RANGE();

    // sigma = zeta * sigma
    constexpr float sgemm_alpha = 1;
    constexpr float sgemm_beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                &sgemm_alpha, d.zeta, n, d.sigma, n,
                                &sgemm_beta, d.temp, n));
    CUDA_CHECK(cudaMemcpy(d.sigma, d.temp, sizeof(float) * n * n, cudaMemcpyDeviceToDevice));
}

void dr_bcg::get_s(cublasHandle_t cublasH, const int m, const int n, DeviceBuffer &d)
{
    NVTX3_FUNC_RANGE();

    // temp = s * zeta'
    constexpr float strmm_alpha = 1;
    CUBLAS_CHECK(cublasStrmm_v2(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, n,
                                &strmm_alpha, d.zeta, n, d.s, m, d.temp, m));

    // s = w + temp
    constexpr float sgeam_alpha = 1;
    constexpr float sgeam_beta = 1;
    CUBLAS_CHECK(cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                             &sgeam_alpha, d.w, m, &sgeam_beta, d.temp, m, d.s, m));
}

void dr_bcg::get_w_zeta(
    cusolverDnHandle_t &cusolverH, cusolverDnParams_t &cusolverParams, cublasHandle_t &cublasH,
    const int m, const int n, DeviceBuffer &d, const float *d_A)
{
    NVTX3_FUNC_RANGE();

    // temp = A * s
    constexpr float alpha_1 = 1;
    constexpr float beta_1 = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                                &alpha_1, d_A, m, d.s, m,
                                &beta_1, d.temp, m));

    // w - temp * xi
    constexpr float alpha_2 = -1;
    constexpr float beta_2 = 1;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                &alpha_2, d.temp, m, d.xi, n,
                                &beta_2, d.w, m));

#ifdef USE_THIN_QR
    thin_qr(cusolverH, cusolverParams, cublasH, d.w, d.zeta, m, n, d.w);
#else
    qr_factorization(cusolverH, cusolverParams, d.w, d.zeta, m, n, d.w);
#endif
}

/**
 * @brief Calculates residual with the following formula: B^(1) - A * X^(1)
 *
 * @param cublasH cuBLAS handle
 * @param d_residual Device workspace for calculation. Result is overwritten to pointed location.
 * @param B Pointer to host memory B
 * @param m The m-value (represents dimensions of square matrix A and length of X and B)
 * @param d_A Pointer to device memory A
 * @param d_X Pointer to device memory X
 */
void dr_bcg::residual(cublasHandle_t &cublasH, float *d_residual, const float *B, const int m, const float *d_A, const float *d_X)
{
    NVTX3_FUNC_RANGE();

    CUDA_CHECK(cudaMemcpy(d_residual, B, sizeof(float) * m, cudaMemcpyDeviceToDevice));

    constexpr float alpha = -1;
    constexpr float beta = 1;
    CUBLAS_CHECK(cublasSgemv_v2(
        cublasH, CUBLAS_OP_N, m, m,
        &alpha, d_A, m, d_X, 1,
        &beta, d_residual, 1));
}

/**
 * @brief Calculates next X guess with the following formula: X_{i+1} = X_{i} + s * xi * sigma
 *
 * @param cublasH cuBLAS handle
 * @param m m dimension
 * @param n n dimension
 * @param d_s Device pointer to s (m x n)
 * @param d_xi Device pointer to xi (n x n)
 * @param d_temp Device pointer to temporary buffer (m x n)
 * @param d_sigma Device pointer to sigma (n x n)
 * @param d_X Device pointer to X (m x n). Result is overwritten to pointed location.
 */
void dr_bcg::get_next_X(cublasHandle_t &cublasH, const int m, const int n, const float *d_s, const float *d_xi, float *d_temp, const float *d_sigma, float *d_X)
{
    NVTX3_FUNC_RANGE();

    constexpr float alpha = 1;
    constexpr float beta = 1;
    CUBLAS_CHECK(cublasStrmm_v2(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                n, n, &alpha, d_sigma, n, d_xi, n, d_temp, n));
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                &alpha, d_s, m, d_temp, n,
                                &beta, d_X, m));
}

/**
 * @brief Compute y = x^T * A * x
 *
 * @param cublasH cuBLAS handle
 * @param m m dimension
 * @param n n dimension
 * @param d_x Device pointer to x (n x m)
 * @param d_A Device pointer to A (m x m)
 * @param d_work Device pointer to workspace
 * @param d_y Device pointer to result y (n x n)
 */
void dr_bcg::quadratic_form(cublasHandle_t &cublasH, const int m, const int n,
                            const float *d_x, const float *d_A,
                            float *d_work, float *d_y)
{
    NVTX3_FUNC_RANGE();

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, m, m,
                                &alpha, d_x, m, d_A, m,
                                &beta, d_work, n));
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, m,
                                &alpha, d_work, n, d_x, m,
                                &beta, d_y, n));
}

/**
 * @brief Computes R = B - AX as GEMM: R = -1.0 * AX + R where R initially contains B.
 *
 * @param cublasH cuBLAS handle
 * @param d_R Device pointer to result R (m x n)
 * @param m m dimension
 * @param n n dimension
 * @param A Host pointer to A (m x m)
 * @param X Host pointer to X (m x n)
 * @param B Host pointer to B (m x n)
 */
void dr_bcg::get_R(cublasHandle_t &cublasH, float *R, const int m, const int n, const float *A, const float *X, const float *B)
{
    NVTX3_FUNC_RANGE();

    constexpr float alpha = -1;
    constexpr float beta = 1;

    CUDA_CHECK(cudaMemcpy(R, B, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                m, n, m,
                                &alpha, A, m, X, m,
                                &beta, R, m));
}

#ifdef USE_THIN_QR

void dr_bcg::qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A)
{
    throw std::runtime_error("qr_factorization not built");
}

#else

/**
 * @brief Computes the QR factorization of matrix A.
 *
 * @param cusolverH cuSOLVER handle
 * @param params Params for the cuSOLVER handle
 * @param Q Pointer to device memory to store Q result in
 * @param R Pointer to device memory to store R result in. Note that the lower triangular still contains householder vectors and must be handled accordingly
 * (e.g. by using trmm in future multiplications using the R factor)
 * @param m m-dimension (leading dimension) of A
 * @param n n-dimension (second dimension) of A
 * @param A The matrix to factorize (device pointer)
 */
void dr_bcg::qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A)
{
    NVTX3_FUNC_RANGE();

    int k = std::min(m, n);
    int info = 0;

    float *d_tau = nullptr;
    int *d_info = nullptr;

    size_t lwork_geqrf_d = 0;
    void *d_work = nullptr;
    size_t lwork_geqrf_h = 0;
    void *h_work = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(float) * k));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpy(Q, A, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

    CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(cusolverH, params, m, n, CUDA_R_32F, Q,
                                               m, CUDA_R_32F, d_tau,
                                               CUDA_R_32F, &lwork_geqrf_d,
                                               &lwork_geqrf_h));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), lwork_geqrf_d));

    if (0 < lwork_geqrf_h)
    {
        h_work = reinterpret_cast<void *>(malloc(lwork_geqrf_h));
        if (h_work == nullptr)
        {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgeqrf(cusolverH, params, m, n, CUDA_R_32F, Q,
                                    m, CUDA_R_32F, d_tau,
                                    CUDA_R_32F, d_work, lwork_geqrf_d, h_work,
                                    lwork_geqrf_h, d_info));
    if (h_work)
    {
        free(h_work); // No longer needed
    }

    const int max_R_col = std::min(m, n);

    copy_upper_triangular(R, Q, m, n);

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }

    // Explicitly compute Q
    int lwork_orgqr = 0;
    CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolverH, m, n, k, Q, m, d_tau, &lwork_orgqr));
    if (lwork_orgqr > lwork_geqrf_d)
    {
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork_orgqr));
    }

    CUSOLVER_CHECK(cusolverDnSorgqr(cusolverH, m, n, k, Q, m, d_tau, reinterpret_cast<float *>(d_work), lwork_orgqr, d_info));

    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
}

#endif

#ifndef USE_THIN_QR

void dr_bcg::thin_qr(
    cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &params,
    cublasHandle_t &cublasH,
    float *Q,
    float *R,
    const int m,
    const int n,
    const float *A)
{
    throw std::runtime_error("thin_qr not built");
}

#else

void dr_bcg::thin_qr(
    cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &params,
    cublasHandle_t &cublasH,
    float *Q,
    float *R,
    const int m,
    const int n,
    const float *A)
{
    NVTX3_FUNC_RANGE();

    // H = M^T * M
    float *d_H = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_H), sizeof(float) * n * n));

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
        &alpha, A, m, A, m,
        &beta, d_H, n));

    // R^T * R = H
    void *h_work = nullptr;
    size_t h_lwork_Xpotrf = 0;
    void *d_work = nullptr;
    size_t d_lwork_Xpotrf = 0;

    int info = 0;
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
        cusolverH, params, CUBLAS_FILL_MODE_UPPER, n,
        CUDA_R_32F, d_H, n,
        CUDA_R_32F, &d_lwork_Xpotrf, &h_lwork_Xpotrf));

    if (h_lwork_Xpotrf > 0)
    {
        h_work = reinterpret_cast<void *>(malloc(h_lwork_Xpotrf));
        if (h_work == nullptr)
        {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_lwork_Xpotrf));

    CUSOLVER_CHECK(cusolverDnXpotrf(
        cusolverH, params, CUBLAS_FILL_MODE_UPPER, n,
        CUDA_R_32F, d_H, n,
        CUDA_R_32F, d_work, d_lwork_Xpotrf, h_work, h_lwork_Xpotrf, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }
    if (info > 0)
    {
        throw std::runtime_error("cusolverDnXpotrf (Cholesky factorization) failed. The smallest leading minor of d_H which is not positive definite is " + std::to_string(info));
    }

    copy_upper_triangular(R, d_H, n, n);

    // Q = M * R^-1
    size_t d_lwork_Xtrtri = 0;
    size_t h_lwork_Xtrtri = 0;
    info = 0;

    CUSOLVER_CHECK(cusolverDnXtrtri_bufferSize(
        cusolverH, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, n,
        CUDA_R_32F, d_H, n,
        &d_lwork_Xtrtri, &h_lwork_Xtrtri));

    if (h_lwork_Xtrtri > h_lwork_Xpotrf)
    {
        if (h_work != nullptr)
        {
            free(h_work);
        }
        h_work = reinterpret_cast<void *>(malloc(h_lwork_Xtrtri));
    }
    if (d_lwork_Xtrtri > d_lwork_Xpotrf)
    {
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_lwork_Xtrtri));
    }

    CUSOLVER_CHECK(cusolverDnXtrtri(
        cusolverH, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, n,
        CUDA_R_32F, d_H, n,
        d_work, d_lwork_Xtrtri, h_work, h_lwork_Xtrtri, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }

    CUBLAS_CHECK(cublasStrmm_v2(
        cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
        &alpha, d_H, n, A, m, Q, m));

    if (h_work != nullptr)
    {
        free(h_work);
    }
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));

    CUDA_CHECK(cudaFree(d_H));
}

#endif

cusolverStatus_t dr_bcg::dr_bcg(
    cusolverDnHandle_t cusolverH,
    cusolverDnParams_t cusolverParams,
    cublasHandle_t cublasH,
    cusparseHandle_t cusparseH,
    cusparseSpMatDescr_t &A,
    cusparseDnMatDescr_t &X,
    cusparseDnMatDescr_t &B,
    float tolerance,
    int max_iterations,
    int *iterations)
{
    NVTX3_FUNC_RANGE();

    int64_t n = 0;
    int64_t s = 0;
    int64_t ld_X = 0;
    float *d_X = nullptr;
    cudaDataType X_dtype;
    cusparseOrder_t X_order;
    CUSPARSE_CHECK(cusparseDnMatGet(X, &n, &s, &ld_X, reinterpret_cast<void **>(&d_X), &X_dtype, &X_order));

    float *d_B = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));

    DeviceBuffer d(n, s);

#ifdef USE_TENSOR_CORES
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TF32_TENSOR_OP_MATH));
#endif

    // We don't include d_R in device buffers because it is only used once at the beginning
    // of the algorithm.
    cusparseDnMatDescr_t R;
    float *d_R;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(float) * n * s));
    CUSPARSE_CHECK(cusparseCreateDnMat(&R, n, s, n, d_R, CUDA_R_32F, CUSPARSE_ORDER_COL));

    // R = B - AX
    get_R(cublasH, cusparseH, R, A, X, B);

#ifdef USE_THIN_QR
    thin_qr(cusolverH, cusolverParams, cublasH, d.w, d.sigma, n, s, d_R);
#else
    qr_factorization(cusolverH, cusolverParams, d.w, d.sigma, n, s, d_R);
#endif

    // R never used later
    CUDA_CHECK(cudaFree(d_R));
    CUSPARSE_CHECK(cusparseDestroyDnMat(R));

    // s = w
    CUDA_CHECK(cudaMemcpy(d.s, d.w, sizeof(float) * n * s, cudaMemcpyDeviceToDevice));

    float B1_norm;
    constexpr int stride = 1;
    CUBLAS_CHECK(cublasSnrm2_v2(cublasH, n, d_B, stride, &B1_norm));

    cusparseDnVecDescr_t r;
    CUSPARSE_CHECK(cusparseCreateDnVec(&r, n, d.residual, CUDA_R_32F));

    int i = 0;
    while (i < max_iterations)
    {
        nvtx3::scoped_range loop{"iteration"};
        ++i;

        // xi = (s' * A * s)^-1
        get_xi(cublasH, cusolverH, cusolverParams, cusparseH, A, n, s, d);

        // X = X + s * xi * sigma
        get_next_X(cublasH, n, s, d.s, d.xi, d.temp, d.sigma, d_X);

        // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
        float relative_residual_norm;
        residual(cusparseH, r, d_B, A, X);

        CUBLAS_CHECK(cublasSnrm2_v2(cublasH, n, d.residual, stride, &relative_residual_norm));
        relative_residual_norm /= B1_norm;

        if (relative_residual_norm < tolerance)
        {
            break;
        }
        else
        {
            nvtx3::scoped_range new_s_and_sigma{"get_new_s_and_sigma"};

            get_w_zeta(cusolverH, cusolverParams, cublasH, cusparseH, n, s, d, A);

            get_s(cublasH, n, s, d);

            get_sigma(cublasH, s, d);
        }
    }

    if (iterations)
    {
        *iterations = i;
    }

    return CUSOLVER_STATUS_SUCCESS;
}

void dr_bcg::get_R(
    cublasHandle_t &cublasH,
    cusparseHandle_t &cusparseH,
    cusparseDnMatDescr_t &R,
    cusparseSpMatDescr_t &A,
    cusparseDnMatDescr_t &X,
    cusparseDnMatDescr_t &B)
{
    NVTX3_FUNC_RANGE();

    constexpr float alpha = -1;
    constexpr float beta = 1;

    float *d_R = nullptr;
    int64_t n = 0;
    int64_t s = 0;
    int64_t ld = 0;
    cudaDataType type;
    cusparseOrder_t order;
    CUSPARSE_CHECK(cusparseDnMatGet(R, &n, &s, &ld, reinterpret_cast<void **>(&d_R), &type, &order));
    float *d_B = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(B, reinterpret_cast<void **>(&d_B)));

    CUDA_CHECK(cudaMemcpy(d_R, d_B, sizeof(float) * n * s, cudaMemcpyDeviceToDevice));

    constexpr cusparseOperation_t transpose = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr cudaDataType compute_type = CUDA_R_32F;
    constexpr cusparseSpMMAlg_t algorithm_type = CUSPARSE_SPMM_ALG_DEFAULT;

    void *buffer = nullptr;
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparseH, transpose, transpose,
        &alpha, A, X, &beta, R,
        compute_type, algorithm_type, &buffer_size));

    if (buffer_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    }

    CUSPARSE_CHECK(cusparseSpMM(
        cusparseH, transpose, transpose,
        &alpha, A, X, &beta, R,
        compute_type, algorithm_type, buffer));

    if (buffer)
    {
        CUDA_CHECK(cudaFree(buffer));
    }
}

void dr_bcg::get_xi(
    cublasHandle_t &cublasH,
    cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &cusolverParams,
    cusparseHandle_t &cusparseH,
    cusparseSpMatDescr_t &A,
    const int n,
    const int s,
    DeviceBuffer &d)
{
    NVTX3_FUNC_RANGE();

    cusparseDnMatDescr_t s_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&s_descr, n, s, n, reinterpret_cast<void *>(d.s), CUDA_R_32F, CUSPARSE_ORDER_COL));

    quadratic_form(cublasH, cusparseH, n, s, s_descr, A, d.temp, d.xi);
    invert_square_matrix(cusolverH, cusolverParams, d.xi, s);
}

void dr_bcg::quadratic_form(
    cublasHandle_t &cublasH,
    cusparseHandle_t &cusparseH,
    const int n,
    const int s,
    cusparseDnMatDescr_t &X,
    cusparseSpMatDescr_t &A,
    float *d_work,
    float *d_y)
{
    NVTX3_FUNC_RANGE();

    constexpr float alpha = 1;
    constexpr float beta = 0;

    constexpr cusparseOperation_t X_transpose = CUSPARSE_OPERATION_TRANSPOSE;
    constexpr cusparseOperation_t A_transpose = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr cudaDataType compute_type = CUDA_R_32F;
    constexpr cusparseSpMMAlg_t algorithm_type = CUSPARSE_SPMM_ALG_DEFAULT;

    // Ax
    cusparseDnMatDescr_t work_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&work_descr, n, s, n, reinterpret_cast<void *>(d_work), CUDA_R_32F, CUSPARSE_ORDER_COL));

    void *buffer = nullptr;
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparseH, X_transpose, A_transpose,
        &alpha, A, X, &beta, work_descr,
        compute_type, algorithm_type, &buffer_size));

    if (buffer_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    }

    CUSPARSE_CHECK(cusparseSpMM(
        cusparseH, X_transpose, A_transpose,
        &alpha, A, X, &beta, work_descr,
        compute_type, algorithm_type, buffer));

    if (buffer)
    {
        CUDA_CHECK(cudaFree(buffer));
    }

    CUSPARSE_CHECK(cusparseDestroyDnMat(work_descr));

    // x^TAx
    float *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));

    CUBLAS_CHECK(cublasSgemm_v2(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N, s, s, n,
        &alpha, d_X, n, d_work, n,
        &beta, d_y, s));
}

void dr_bcg::residual(
    cusparseHandle_t &cusparseH,
    cusparseDnVecDescr_t &residual,
    const float *B,
    cusparseSpMatDescr_t &A,
    cusparseDnMatDescr_t &X)
{
    NVTX3_FUNC_RANGE();

    int64_t n = 0;
    float *d_residual = nullptr;
    cudaDataType residual_dtype;
    CUSPARSE_CHECK(cusparseDnVecGet(residual, &n, reinterpret_cast<void **>(&d_residual), &residual_dtype));

    CUDA_CHECK(cudaMemcpy(d_residual, B, sizeof(float) * n, cudaMemcpyDeviceToDevice));

    float *d_X = nullptr;
    CUSPARSE_CHECK(cusparseDnMatGetValues(X, reinterpret_cast<void **>(&d_X)));
    cusparseDnVecDescr_t X_1;
    CUSPARSE_CHECK(cusparseCreateDnVec(&X_1, n, d_X, CUDA_R_32F));

    constexpr float alpha = -1;
    constexpr float beta = 1;

    void *buffer = nullptr;
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A, X_1, &beta, residual,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));

    if (buffer_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    }

    CUSPARSE_CHECK(cusparseSpMV(
        cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A, X_1, &beta, residual,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    if (buffer)
    {
        CUDA_CHECK(cudaFree(buffer));
    }

    CUSPARSE_CHECK(cusparseDestroyDnVec(X_1));
}

void dr_bcg::get_w_zeta(
    cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &cusolverParams,
    cublasHandle_t &cublasH,
    cusparseHandle_t &cusparseH,
    const int m,
    const int n,
    DeviceBuffer &d,
    cusparseSpMatDescr_t &A)
{
    NVTX3_FUNC_RANGE();

    std::cerr << "TODO: implement get_w_zeta" << std::endl;

    //     // temp = A * s
    //     constexpr float alpha_1 = 1;
    //     constexpr float beta_1 = 0;
    //     CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
    //                                 &alpha_1, d_A, m, d.s, m,
    //                                 &beta_1, d.temp, m));

    //     // w - temp * xi
    //     constexpr float alpha_2 = -1;
    //     constexpr float beta_2 = 1;
    //     CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
    //                                 &alpha_2, d.temp, m, d.xi, n,
    //                                 &beta_2, d.w, m));

    // #ifdef USE_THIN_QR
    //     thin_qr(cusolverH, cusolverParams, cublasH, d.w, d.zeta, m, n, d.w);
    // #else
    //     qr_factorization(cusolverH, cusolverParams, d.w, d.zeta, m, n, d.w);
    // #endif
}