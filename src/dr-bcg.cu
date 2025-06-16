#include <iostream>
#include <vector>

#include "dr_bcg/dr-bcg.h"
#include "dr_bcg/helper.h"

namespace dr_bcg
{
    int dr_bcg(
        float *A,
        const int m,
        const int n,
        const float *X,
        const float *B,
        const float tolerance,
        const int max_iterations)
    {
        cublasHandle_t cublasH;
        CUBLAS_CHECK(cublasCreate(&cublasH));

        // R = B - AX
        std::vector<float> R(m * n);
        float *d_R;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(float) * R.size()));
        get_R(cublasH, d_R, m, n, A, X, B);
        CUDA_CHECK(cudaMemcpy(R.data(), d_R, sizeof(float) * R.size(), cudaMemcpyDeviceToHost));

        cusolverDnHandle_t cusolverH = NULL;
        cusolverDnParams_t cusolverParams = NULL;

        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

        // [w, sigma] = qr(R)
        std::vector<float> w(m * n);
        std::vector<float> sigma(n * n);

        float *d_w;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_w), sizeof(float) * w.size()));
        CUDA_CHECK(cudaMemcpy(d_w, w.data(), sizeof(float) * w.size(), cudaMemcpyHostToDevice));
        float *d_sigma;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_sigma), sizeof(float) * sigma.size()));
        CUDA_CHECK(cudaMemcpy(d_sigma, sigma.data(), sizeof(float) * sigma.size(), cudaMemcpyHostToDevice));

        qr_factorization(cusolverH, cusolverParams, d_w, d_sigma, m, n, d_R);

        CUDA_CHECK(cudaMemcpy(w.data(), d_w, sizeof(float) * w.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sigma.data(), d_sigma, sizeof(float) * sigma.size(), cudaMemcpyDeviceToHost));

        std::vector<float> s = std::move(w);

        float alpha = 1.0;
        float beta = 0.0;
        int iterations;

        float *d_A;
        float *d_s;
        float *d_temp;
        float *d_xi;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_s), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), sizeof(float) * n * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_xi), sizeof(float) * n * n));

        CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(float) * m * m, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s, s.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));

        float *d_X;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * m * n));
        CUDA_CHECK(cudaMemcpy(d_X, X, sizeof(float) * m * n, cudaMemcpyHostToDevice));

        for (iterations = 1; iterations < 2; iterations++)
        {
            // xi = (s' * A * s)^-1
            quadratic_form(cublasH, m, n, alpha, d_s, d_A, beta, d_temp, d_xi);
            invert_spd(cusolverH, cusolverParams, d_xi, n);

            // X = X + s * xi * sigma
            next_X(cublasH, m, n, d_s, d_xi, d_temp, d_sigma, d_X);

            std::vector<float> debug_X(m * n);
            CUDA_CHECK(cudaMemcpy(debug_X.data(), d_X, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
            std::cout << "DEBUG: X" << std::endl;
            print_matrix(debug_X.data(), m, n);
        }

        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_w));
        CUDA_CHECK(cudaFree(d_sigma));
        CUDA_CHECK(cudaFree(d_X));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_s));
        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaFree(d_xi));

        CUBLAS_CHECK(cublasDestroy_v2(cublasH));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

        return iterations;
    }

    /// @brief Calculates X_{i+1} = X_{i} + s * xi * sigma
    /// @param d_X (device memory pointer) X_{i}. Result is overwritten to pointed location
    void next_X(cublasHandle_t cublasH, const int m, const int n, float *d_s, float *d_xi, float *d_temp, float *d_sigma, float *d_X)
    {
        float alpha = 1;
        float beta = 0;
        CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                    &alpha, d_s, m, d_xi, n,
                                    &beta, d_temp, m));
        beta = 1;
        CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                    &alpha, d_temp, m, d_sigma, n,
                                    &beta, d_X, m));
    }

    /// @brief Compute y = x^T * A * x
    void quadratic_form(cublasHandle_t cublasH, const int m, const int n,
                        float &alpha, float *d_x, float *d_A,
                        float &beta, float *d_work, float *d_y)
    {
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

    void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A)
    {
        int k = std::min(m, n);
        std::vector<float> tau(k, 0);
        int info = 0;

        float *d_tau = nullptr;
        int *d_info = nullptr;

        size_t lwork_geqrf_d = 0;
        void *d_work = nullptr;
        size_t lwork_geqrf_h = 0;
        void *h_work = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(float) * tau.size()));
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
        free(h_work); // No longer needed

        CUDA_CHECK(cudaMemcpy(R, Q, sizeof(float) * n * n, cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMemcpy(tau.data(), d_tau, sizeof(float) * tau.size(), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            std::printf("%d-th parameter is wrong \n", -info);
            exit(1);
        }

        // Explicitly compute Q
        int lwork_orgqr = 0;
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolverH, m, n, k, Q, m, d_tau, &lwork_orgqr));
        CUSOLVER_CHECK(cusolverDnSorgqr(cusolverH, m, n, k, Q, m, d_tau, reinterpret_cast<float *>(d_work), lwork_orgqr, d_info));

        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_tau));
        CUDA_CHECK(cudaFree(d_work));
    }

    /// @brief Computes the inverse of a matrix using Cholesky factorization
    /// @param A (device memory pointer) the symmetric positive definite matrix to invert. Result is overwritten to pointed location.
    void invert_spd(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *A, const int64_t n)
    {
        size_t workspaceInBytesOnDevice = 0;
        void *d_work = nullptr;
        size_t workspaceInBytesOnHost = 0;
        void *h_work = nullptr;

        int info = 0;
        int *d_info = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

        CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverH, params, CUBLAS_FILL_MODE_LOWER,
                                                   n, CUDA_R_32F, A, n, CUDA_R_32F,
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
                                        n, CUDA_R_32F, A, n,
                                        CUDA_R_32F, d_work, workspaceInBytesOnDevice,
                                        h_work, workspaceInBytesOnHost, d_info));

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            std::fprintf(stderr, "%d-th parameter is wrong \n", -info);
            exit(1);
        }

        // TODO: Parallelize this
        std::vector<float> I(n * n, 0);
        float *d_I = nullptr;
        for (int i = 0; i < n; i++)
        {
            I.at(i * n + i) = 1;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_I), sizeof(float) * n * n));
        CUDA_CHECK(cudaMemcpy(d_I, I.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice));

        CUSOLVER_CHECK(cusolverDnXpotrs(
            cusolverH, params, CUBLAS_FILL_MODE_LOWER,
            n, n, CUDA_R_32F, A, n, CUDA_R_32F, d_I, n, d_info));
        if (0 > info)
        {
            std::fprintf(stderr, "%d-th parameter is wrong \n", -info);
            exit(1);
        }

        CUDA_CHECK(cudaMemcpy(A, d_I, sizeof(float) * n * n, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_I));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_work));
    }
}
