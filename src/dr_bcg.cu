#include <iostream>
#include <vector>
#include <tuple>
#include <string>

#include <nvtx3/nvtx3.hpp>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

/**
 * @brief Device pointers for reused device buffers.
 *
 * This struct manages device memory for all buffers used in the DR-BCG algorithm.
 * It handles allocation and deallocation of all required device arrays.
 */
struct DeviceBuffer
{
    float *w = nullptr;        ///< Device pointer for matrix w (m x n)
    float *sigma = nullptr;    ///< Device pointer for matrix sigma (n x n)
    float *s = nullptr;        ///< Device pointer for matrix s (m x n)
    float *xi = nullptr;       ///< Device pointer for matrix xi (n x n)
    float *zeta = nullptr;     ///< Device pointer for matrix zeta (n x n)
    float *temp = nullptr;     ///< Device pointer for temporary matrix (m x n)
    float *residual = nullptr; ///< Device pointer for residual vector (m)

    /**
     * @brief Constructor. Allocates all device buffers.
     * @param m m dimension
     * @param n n dimension
     */
    DeviceBuffer(int m, int n)
    {
        allocate(m, n);
    }

    /**
     * @brief Destructor. Frees all allocated device memory.
     */
    ~DeviceBuffer()
    {
        deallocate();
    }

    /**
     * @brief Allocates device memory for all buffers.
     * @param m m dimension
     * @param n n dimension
     */
    void allocate(int m, int n)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&w), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sigma), sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&xi), sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&zeta), sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temp), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&residual), sizeof(float) * m));
    }

    /**
     * @brief Deallocates all device memory.
     */
    void deallocate()
    {
        CUDA_CHECK(cudaFree(w));
        CUDA_CHECK(cudaFree(sigma));
        CUDA_CHECK(cudaFree(s));
        CUDA_CHECK(cudaFree(xi));
        CUDA_CHECK(cudaFree(zeta));
        CUDA_CHECK(cudaFree(temp));
        CUDA_CHECK(cudaFree(residual));
    }
};

/**
 * @brief CUDA kernel to symmetrize a square matrix in-place.
 *
 * Copies the lower triangle to the upper triangle to ensure symmetry.
 * @param A Pointer to device matrix (n x n)
 * @param n Matrix dimension
 */
__global__ void symmetrize_matrix(float *A, const int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < col && row < n && col < n)
    {
        A[col * n + row] = A[row * n + col];
    }
}

/**
 * @brief CUDA kernel to copy upper triangular of a matrix stored in column-major order.
 *
 * @param dst Pointer to destination device matrix (n x n)
 * @param src Pointer to source device matrix (n x n)
 * @param n Matrix dimension
 */
__global__ void copy_upper_triangular(float *dst, float *src, const int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col <= row && row < n && col < n)
    {
        dst[row * n + col] = src[row * n + col];
    }
    else
    {
        dst[row * n + col] = 0;
    }
}

namespace dr_bcg
{
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
    std::tuple<std::vector<float>, int> dr_bcg(
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
        int *iterations)
    {
        NVTX3_FUNC_RANGE();

        DeviceBuffer d(m, n);

        // We don't include d_R in device buffers because it is only used once at the beginning
        // of the algorithm.
        float *d_R;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(float) * m * n));

        // R = B - AX
        get_R(cublasH, d_R, m, n, A, X, B);

        // [w, sigma] = qr(R)
        qr_factorization(cusolverH, cusolverParams, d.w, d.sigma, m, n, d_R);

        CUDA_CHECK(cudaFree(d_R)); // Never used later

        // s = w
        CUDA_CHECK(cudaMemcpy(d.s, d.w, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

        float B1_norm;
        CUBLAS_CHECK(cublasSnrm2_v2(cublasH, m, B, 1, &B1_norm));

        *iterations = 0;
        while (*iterations < max_iterations)
        {
            nvtx3::scoped_range loop{"iteration_" + std::to_string(*iterations)};

            (*iterations)++;

            // xi = (s' * A * s)^-1
            quadratic_form(cublasH, m, n, d.s, A, d.temp, d.xi);

            invert_square_matrix(cusolverH, cusolverParams, d.xi, n);

            // X = X + s * xi * sigma
            next_X(cublasH, m, n, d.s, d.xi, d.temp, d.sigma, X);

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

                // temp = A * s
                float alpha = 1;
                float beta = 0;
                CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                                            &alpha, A, m, d.s, m,
                                            &beta, d.temp, m));

                // w - temp * xi
                alpha = -1;
                beta = 1;
                CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                            &alpha, d.temp, m, d.xi, n,
                                            &beta, d.w, m));

                qr_factorization(cusolverH, cusolverParams, d.w, d.zeta, m, n, d.w);

                // temp = s * zeta'
                alpha = 1;
                CUBLAS_CHECK(cublasStrmm_v2(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, n,
                                            &alpha, d.zeta, n, d.s, m, d.temp, m));

                // s = w + temp
                beta = 1;
                CUBLAS_CHECK(cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                                         &alpha, d.w, m, &beta, d.temp, m, d.s, m));

                // sigma = zeta * sigma
                beta = 0;
                CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                            &alpha, d.zeta, n, d.sigma, n,
                                            &beta, d.temp, n));
                CUDA_CHECK(cudaMemcpy(d.sigma, d.temp, sizeof(float) * n * n, cudaMemcpyDeviceToDevice));
            }
        }

        return CUSOLVER_STATUS_SUCCESS;
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
    void residual(cublasHandle_t &cublasH, float *d_residual, const float *B, const int m, const float *d_A, const float *d_X)
    {
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
    void next_X(cublasHandle_t &cublasH, const int m, const int n, const float *d_s, const float *d_xi, float *d_temp, const float *d_sigma, float *d_X)
    {
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
    void quadratic_form(cublasHandle_t &cublasH, const int m, const int n,
                        const float *d_x, const float *d_A,
                        float *d_work, float *d_y)
    {
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
    void get_R(cublasHandle_t &cublasH, float *R, const int m, const int n, const float *A, const float *X, const float *B)
    {
        constexpr float alpha = -1;
        constexpr float beta = 1;

        CUDA_CHECK(cudaMemcpy(R, B, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

        CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, n, m,
                                    &alpha, A, m, X, m,
                                    &beta, R, m));
    }

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
    void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A)
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

        constexpr int block_n = 16;
        dim3 block_dim(block_n, block_n);
        dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
        copy_upper_triangular<<<grid_dim, block_dim>>>(R, Q, n);

        for (int col = 0; col < max_R_col; col++)
        {
            CUDA_CHECK(cudaMemcpy(R + col * n, Q + col * m, sizeof(float) * (col + 1), cudaMemcpyDeviceToDevice));
        }

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

    /**
     * @brief Computes the inverse of a matrix using Cholesky factorization.
     *
     * @param cusolverH cuSOLVER handle
     * @param params cuSOLVER params
     * @param d_A Device pointer to the symmetric positive definite matrix to invert. Result is overwritten to pointed location.
     * @param n Matrix dimension
     */
    void invert_square_matrix(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *d_A, const int n)
    {
        NVTX3_FUNC_RANGE();

        // LU Decomposition
        size_t workspaceInBytesOnDevice = 0;
        void *d_work = nullptr;
        size_t workspaceInBytesOnHost = 0;
        void *h_work = nullptr;

        int info = 0;
        int *d_info = nullptr;

        std::vector<int64_t> h_Ipiv(n, 0);
        int64_t *d_Ipiv = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * h_Ipiv.size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

        CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n,
                                                   CUDA_R_32F, d_A, n, CUDA_R_32F,
                                                   &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
        if (0 < workspaceInBytesOnHost)
        {
            h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
            if (h_work == nullptr)
            {
                throw std::runtime_error("Error: h_work not allocated.");
            }
        }

        CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, n, n,
                                        CUDA_R_32F, d_A, n, d_Ipiv, CUDA_R_32F,
                                        d_work, workspaceInBytesOnDevice,
                                        h_work, workspaceInBytesOnHost, d_info));

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
        }

        CUDA_CHECK(cudaFree(d_work));
        free(h_work);

        // Solve A * X = I for inverse
        std::vector<float> h_I(n * n, 0);
        float *d_I = nullptr;

        for (int i = 0; i < n; i++)
        {
            h_I.at(i * n + i) = 1;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_I), sizeof(float) * h_I.size()));
        CUDA_CHECK(cudaMemcpy(d_I, h_I.data(), sizeof(float) * h_I.size(), cudaMemcpyHostToDevice));

        CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n,
                                        CUDA_R_32F, d_A, n, d_Ipiv, CUDA_R_32F,
                                        d_I, n, d_info));

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
        }

        CUDA_CHECK(cudaMemcpy(d_A, d_I, sizeof(float) * h_I.size(), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaFree(d_I));

        CUDA_CHECK(cudaFree(d_Ipiv));
        CUDA_CHECK(cudaFree(d_info));
    }
}
