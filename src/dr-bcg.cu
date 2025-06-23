#include <iostream>
#include <vector>
#include <string>

#include "dr_bcg/dr-bcg.h"
#include "dr_bcg/helper.h"

// Device pointers for reused device buffers
struct DeviceBuffer
{
    float *A = nullptr;
    float *X = nullptr;
    float *w = nullptr;
    float *sigma = nullptr;
    float *s = nullptr;
    float *xi = nullptr;
    float *zeta = nullptr;
    float *temp = nullptr;
    float *residual = nullptr;

    DeviceBuffer(int m, int n)
    {
        allocate(m, n);
    }

    ~DeviceBuffer()
    {
        deallocate();
    }

    void allocate(int m, int n)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A), sizeof(float) * m * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&X), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&w), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sigma), sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&xi), sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&zeta), sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temp), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&residual), sizeof(float) * m));
    }

    void deallocate()
    {
        CUDA_CHECK(cudaFree(A));
        CUDA_CHECK(cudaFree(X));
        CUDA_CHECK(cudaFree(w));
        CUDA_CHECK(cudaFree(sigma));
        CUDA_CHECK(cudaFree(s));
        CUDA_CHECK(cudaFree(xi));
        CUDA_CHECK(cudaFree(zeta));
        CUDA_CHECK(cudaFree(temp));
        CUDA_CHECK(cudaFree(residual));
    }
};

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
    int dr_bcg(
        const float *A,
        const int m,
        const int n,
        float *X,
        const float *B,
        const float tolerance,
        const int max_iterations)
    {
        cublasHandle_t cublasH;
        CUBLAS_CHECK(cublasCreate(&cublasH));

        cusolverDnHandle_t cusolverH = NULL;
        cusolverDnParams_t cusolverParams = NULL;
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

        DeviceBuffer d(m, n);
        CUDA_CHECK(cudaMemcpy(d.A, A, sizeof(float) * m * m, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.X, X, sizeof(float) * m * n, cudaMemcpyHostToDevice));

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
        float *d_B1 = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B1), sizeof(float) * m));
        CUDA_CHECK(cudaMemcpy(d_B1, B, sizeof(float) * m, cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSnrm2_v2(cublasH, m, d_B1, 1, &B1_norm));
        CUDA_CHECK(cudaFree(d_B1));

        int iterations;
        for (iterations = 1; iterations <= max_iterations; iterations++)
        {
            // xi = (s' * A * s)^-1
            quadratic_form(cublasH, m, n, d.s, d.A, d.temp, d.xi);
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

        CUBLAS_CHECK(cublasDestroy_v2(cublasH));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
        CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverParams));

        return iterations;
    }

    /// @brief Calculates residual with the following formula: B^(1) - A * X^(1)
    /// @param cublasH cuBLAS handle
    /// @param d_residual device workspace for calculation. Result is overwritten to pointed location
    /// @param B pointer to host memory B
    /// @param m the m-value (represents dimensions of square matrix A and length of X and B)
    /// @param d_A pointer to device memory A
    /// @param d_X pointer to device memory X
    void residual(cublasHandle_t &cublasH, float *d_residual, const float *B, const int m, const float *d_A, const float *d_X)
    {
        CUDA_CHECK(cudaMemcpy(d_residual, B, sizeof(float) * m, cudaMemcpyHostToDevice));

        constexpr float alpha = -1;
        constexpr float beta = 1;
        CUBLAS_CHECK(cublasSgemv_v2(
            cublasH, CUBLAS_OP_N, m, m,
            &alpha, d_A, m, d_X, 1,
            &beta, d_residual, 1));
    }

    /// @brief Calculates X_{i+1} = X_{i} + s * xi * sigma
    /// @param d_X (device memory pointer) X_{i}. Result is overwritten to pointed location
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

    /// @brief Compute y = x^T * A * x
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

    // R = B - AX as GEMM:
    // R = -1.0 * AX + R where R initially contains B
    void get_R(cublasHandle_t &cublasH, float *d_R, const int m, const int n, const float *A, const float *X, const float *B)
    {
        constexpr float alpha = -1;
        constexpr float beta = 1;

        float *d_A = nullptr;
        float *d_X = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * m * n));

        CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(float) * m * m, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_X, X, sizeof(float) * m * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_R, B, sizeof(float) * m * n, cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, n, m,
                                    &alpha, d_A, m, d_X, m,
                                    &beta, d_R, m));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_X));
    }

    /// @brief Computes the QR factorization of matrix A
    /// @param cusolverH cuSOLVER handle
    /// @param params params for the cuSOLVER handle
    /// @param Q pointer to device memory to store Q result in
    /// @param R pointer to device memory to store R result in. Note that the lower triangular still contains householder vectors and must be handled accordingly
    /// (e.g. by using trmm in future multiplications using the R factor)
    /// @param m m-dimension (leading dimension) of A
    /// @param n n-dimension (second dimension) of A
    /// @param A the matrix to factorize
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

    /// @brief Computes the inverse of a matrix using Cholesky factorization
    /// @param A (device memory pointer) the symmetric positive definite matrix to invert. Result is overwritten to pointed location.
    void invert_spd(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *d_A, const int n)
    {
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

        CUSOLVER_CHECK(cusolverDnXpotrf(cusolverH, params, CUBLAS_FILL_MODE_LOWER,
                                        n, CUDA_R_32F, d_A, n,
                                        CUDA_R_32F, d_work, workspaceInBytesOnDevice,
                                        h_work, workspaceInBytesOnHost, d_info));

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
        }
        CUDA_CHECK(cudaFree(d_work));

        float *d_work_Spotri = nullptr;
        int lwork_Spotri = 0;
        info = 0;
        CUSOLVER_CHECK(cusolverDnSpotri_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, n,
                                                   d_A, n, &lwork_Spotri));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work_Spotri), lwork_Spotri));
        CUSOLVER_CHECK(cusolverDnSpotri(cusolverH, CUBLAS_FILL_MODE_LOWER, n,
                                        d_A, n, d_work_Spotri, lwork_Spotri, d_info));

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
        }

        constexpr int block_n = 16;
        dim3 block_dim(block_n, block_n);
        dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
        symmetrize_matrix<<<grid_dim, block_dim>>>(d_A, n);

        CUDA_CHECK(cudaFree(d_work_Spotri));

        CUDA_CHECK(cudaFree(d_info));
    }
}
