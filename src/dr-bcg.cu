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
        int iterations = 0;

        cublasHandle_t cublasH;
        CUBLAS_CHECK(cublasCreate(&cublasH));

        // R = B - AX
        std::vector<float> R(m * n);
        get_R(cublasH, R.data(), m, n, A, X, B);

        std::cout << "\nAfter R = B - AX\n"
                  << std::endl;
        std::cout << "A:" << std::endl;
        print_matrix(A, m, m);
        std::cout << "X:" << std::endl;
        print_matrix(X, m, n);
        std::cout << "B:" << std::endl;
        print_matrix(B, m, n);
        std::cout << "R:" << std::endl;
        print_matrix(R.data(), m, n);

        cusolverDnHandle_t cusolverH = NULL;
        cusolverDnParams_t cusolverParams = NULL;

        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

        // [w, sigma] = qr(R)
        std::vector<float> w(m * n);
        std::vector<float> sigma(n * n);
        qr_factorization(cusolverH, cusolverParams, w.data(), sigma.data(), m, n, R.data());

        std::cout << "\nAfter [w, sigma] = qr(R)\n"
                  << std::endl;
        std::cout << "w:" << std::endl;
        print_matrix(w.data(), m, n);
        std::cout << "sigma:" << std::endl;
        print_matrix(sigma.data(), n, n);

        CUBLAS_CHECK(cublasDestroy_v2(cublasH));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

        for (int k = 0; k < max_iterations; k++)
        {
            iterations++;
        }

        return iterations;
    }

    // R = B - AX as GEMM:
    // R = -1.0 * AX + R where R initially contains B
    void get_R(cublasHandle_t &cublasH, float *h_R, const int m, const int n, const float *A, const float *X, const float *B)
    {
        constexpr float alpha = -1;
        constexpr float beta = 1;

        float *d_A = nullptr;
        float *d_X = nullptr;
        float *d_R = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(float) * m * n));

        CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(float) * m * m, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_X, X, sizeof(float) * m * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_R, B, sizeof(float) * m * n, cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, n, m,
                                    &alpha, d_A, m, d_X, m,
                                    &beta, d_R, m));

        CUDA_CHECK(cudaMemcpy(h_R, d_R, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_R));
    }

    void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A)
    {
        int k = std::min(m, n);
        std::vector<float> tau(k, 0);
        int info = 0;

        float *d_A = nullptr;
        float *d_tau = nullptr;
        int *d_info = nullptr;

        size_t lwork_geqrf_d = 0;
        void *d_work = nullptr;
        size_t lwork_geqrf_h = 0;
        void *h_work = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(float) * tau.size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice));

        CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(cusolverH, params, m, n, CUDA_R_32F, d_A,
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

        CUSOLVER_CHECK(cusolverDnXgeqrf(cusolverH, params, m, n, CUDA_R_32F, d_A,
                                        m, CUDA_R_32F, d_tau,
                                        CUDA_R_32F, d_work, lwork_geqrf_d, h_work,
                                        lwork_geqrf_h, d_info));
        free(h_work); // No longer needed

        // Copy R to host (stored in upper triangular)
        CUDA_CHECK(cudaMemcpy(R, d_A, sizeof(float) * n * n, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(tau.data(), d_tau, sizeof(float) * tau.size(), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (0 > info)
        {
            std::printf("%d-th parameter is wrong \n", -info);
            exit(1);
        }

        // Explicitly compute Q
        int lwork_orgqr = 0;
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolverH, m, n, k, d_A, m, d_tau, &lwork_orgqr));
        CUSOLVER_CHECK(cusolverDnSorgqr(cusolverH, m, n, k, d_A, m, d_tau, reinterpret_cast<float *>(d_work), lwork_orgqr, d_info));

        // Copy Q to host
        CUDA_CHECK(cudaMemcpy(Q, d_A, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_tau));
        CUDA_CHECK(cudaFree(d_work));
    }
}
