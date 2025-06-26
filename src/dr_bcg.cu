#include <iostream>
#include <vector>
#include <tuple>
#include <string>

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
    const float *A = nullptr;  ///< Device pointer for matrix A (m x m)
    float *X = nullptr;        ///< Device pointer for matrix X (m x n)
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

namespace dr_bcg
{
    // Helper function to check for NaNs in a device array
    void check_nan(const float *d_arr, size_t size, std::string step)
    {
        std::vector<float> h_arr(size);
        CUDA_CHECK(cudaMemcpy(h_arr.data(), d_arr, sizeof(float) * size, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < size; ++i)
        {
            if (std::isnan(h_arr[i]))
            {
                throw std::runtime_error("NaN detected after step: " + step);
            }
        }
    }

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
        DeviceBuffer d(m, n);
        d.A = A;
        d.X = X;

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
            (*iterations)++;

            // xi = (s' * A * s)^-1
            quadratic_form(cublasH, m, n, d.s, d.A, d.temp, d.xi);
            check_nan(d.xi, n * n, "quadratic_form (xi before invert)");

            print_device_matrix(d.xi, n, n);

            invert_spd(cusolverH, cusolverParams, d.xi, n);

            // X = X + s * xi * sigma
            next_X(cublasH, m, n, d.s, d.xi, d.temp, d.sigma, d.X);

            // norm(B(:,1) - A * X(:,1)) / norm(B(:,1))
            float relative_residual_norm;
            residual(cublasH, d.residual, B, m, d.A, d.X);

            CUBLAS_CHECK(cublasSnrm2_v2(cublasH, m, d.residual, 1, &relative_residual_norm));
            relative_residual_norm /= B1_norm;

            if (relative_residual_norm < tolerance)
            {
                break;
            }
            else
            {
                // temp = A * s
                float alpha = 1;
                float beta = 0;
                CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                                            &alpha, d.A, m, d.s, m,
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

        CUDA_CHECK(cudaMemcpy(X, d.X, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

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
    void invert_spd(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *d_A, const int n)
    {
        // Cholesky Decomposition
        size_t workspaceInBytesOnDevice = 0;
        void *d_work = nullptr;
        size_t workspaceInBytesOnHost = 0;
        void *h_work = nullptr;

        int info = 0;
        int *d_info = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

        CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverH, params, CUBLAS_FILL_MODE_LOWER,
                                                   n, CUDA_R_32F, d_A, n, CUDA_R_32F,
                                                   &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
        if (0 < workspaceInBytesOnHost)
        {
            h_work = reinterpret_cast<void *>(malloc(sizeof(float) * workspaceInBytesOnHost));
            if (h_work == nullptr)
            {
                throw std::runtime_error("Error: h_work not allocated.");
            }
        }

        std::cout << "\nBefore Xpotrf" << std::endl;
        print_device_matrix(d_A, n, n);

        CUSOLVER_CHECK(cusolverDnXpotrf(cusolverH, params, CUBLAS_FILL_MODE_LOWER,
                                        n, CUDA_R_32F, d_A, n,
                                        CUDA_R_32F, d_work, workspaceInBytesOnDevice,
                                        h_work, workspaceInBytesOnHost, d_info));

        std::cout << "After Xpotrf" << std::endl;
        print_device_matrix(d_A, n, n);

        check_nan(d_A, n * n, "invert_spd::Xpotrf");

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong.");
        }
        if (info > 0)
        {
            throw std::runtime_error("Smallest leading minor " + std::to_string(info) + " is not positive definite.");
        }
        CUDA_CHECK(cudaFree(d_work));

        // Inversion
        float *d_work_Spotri = nullptr;
        int lwork_Spotri = 0;
        info = 0;
        CUSOLVER_CHECK(cusolverDnSpotri_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, n,
                                                   d_A, n, &lwork_Spotri));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work_Spotri), lwork_Spotri));
        CUSOLVER_CHECK(cusolverDnSpotri(cusolverH, CUBLAS_FILL_MODE_LOWER, n,
                                        d_A, n, d_work_Spotri, lwork_Spotri, d_info));

        check_nan(d_A, n * n, "invert_spd::Spotri");

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong.");
        }
        if (info > 0)
        {
            throw std::runtime_error("Leading minor of order " + std::to_string(info) + " is zero.");
        }

        constexpr int block_n = 16;
        dim3 block_dim(block_n, block_n);
        dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
        symmetrize_matrix<<<grid_dim, block_dim>>>(d_A, n);

        CUDA_CHECK(cudaFree(d_work_Spotri));

        CUDA_CHECK(cudaFree(d_info));
    }
}
