#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "helper.h"
#include "dr-bcg.h"

void print_matrix(const float *mat, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%6.3f ", mat[i * cols + j]);
        }
        std::cout << std::endl;
    }
}

/*
 * function [X_final, iterations] = DR_BCG(A, B, X, tol, maxit)
 *     iterations = 0;
 *     R = B - A * X;
 *     [w, sigma] = qr(R,'econ');
 *     s = w;

 *     for k = 1:maxit
 *         iterations = iterations + 1;
 *         xi = (s' * A * s)^-1;
 *         X = X + s * xi * sigma;
 *         if (norm(B(:,1) - A * X(:,1)) / norm(B(:,1))) < tol
 *             break
 *         else
 *             [w, zeta] = qr(w - A * s * xi,'econ');
 *             s = w + s * zeta';
 *             sigma = zeta * sigma;
 *         end
 *     end
 *     X_final = X;
 * end
 */
int dr_bcg(
    float *A,
    const int n,
    const float *x,
    const float *b,
    const float tolerance,
    const int max_iterations)
{
    int iterations = 0;

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    // r = b - Ax as GEMM:
    // r = -1.0 * Ax + r where r initially contains b
    const float alpha = -1;
    const float beta = 1;

    float *h_r = (float *)malloc(n * sizeof(float));
    float *d_A = nullptr;
    float *d_x = nullptr;
    float *d_r = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * n * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_r), sizeof(float) * n));

    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, b, sizeof(float) * n, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasSgemv(
        cublasH,
        CUBLAS_OP_N,
        n,
        n,
        &alpha,
        d_A, n,
        d_x, 1,
        &beta,
        d_r, 1));

    CUDA_CHECK(cudaMemcpy(h_r, d_r, sizeof(float) * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));

    std::cout << "\nAfter r = b - Ax\n"
              << std::endl;
    std::cout << "A:" << std::endl;
    print_matrix(A, n, n);
    std::cout << "x:" << std::endl;
    print_matrix(x, n, 1);
    std::cout << "r:" << std::endl;
    print_matrix(h_r, n, 1);

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
    free(h_r);

    std::cout << "[INFO]Starting QR procedure [w, sigma] = qr(r)" << std::endl;
    std::vector<float> w(n * n);
    std::vector<float> sigma(n * n);
    qr_decomposition(w.data(), sigma.data(), n, A, b);

    std::cout << "\nAfter [w, sigma] = qr(r)\n"
              << std::endl;
    std::cout << "w:" << std::endl;
    print_matrix(w.data(), n, n);
    std::cout << "sigma:" << std::endl;
    print_matrix(sigma.data(), n, n);

    return iterations;
}

void qr_decomposition(float *q, float *r, const int n, float *A, const float *b)
{
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t params = NULL;

    using data_type = float;

    std::vector<data_type> tau(n, 0);
    int info = 0;

    data_type *d_A = nullptr;
    data_type *d_b = nullptr;
    data_type *d_tau = nullptr;
    int *d_info = nullptr;

    size_t lwork_geqrf_d = 0;
    void *d_work = nullptr;
    size_t lwork_geqrf_h = 0;
    void *h_work = nullptr;

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * n * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(data_type) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(data_type) * tau.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(data_type) * n * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(data_type) * n, cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(cusolverH, params, n, n, CUDA_R_32F, d_A,
                                               n, CUDA_R_32F, d_tau,
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

    CUSOLVER_CHECK(cusolverDnXgeqrf(cusolverH, params, n, n, CUDA_R_32F, d_A,
                                    n, CUDA_R_32F, d_tau,
                                    CUDA_R_32F, d_work, lwork_geqrf_d, h_work,
                                    lwork_geqrf_h, d_info));

    // Copy R (stored in upper triangular)
    CUDA_CHECK(cudaMemcpy(r, d_A, sizeof(data_type) * n * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(tau.data(), d_tau, sizeof(data_type) * tau.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 > info)
    {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(A, d_A, sizeof(data_type) * n * n, cudaMemcpyDeviceToHost));

    // Explicitly compute Q
    int lwork_orgqr = 0;
    CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolverH, n, n, n, d_A, n, d_tau, &lwork_orgqr));
    CUSOLVER_CHECK(cusolverDnSorgqr(cusolverH, n, n, n, d_A, n, d_tau, reinterpret_cast<float *>(d_work), lwork_orgqr, d_info));

    // Copy Q
    CUDA_CHECK(cudaMemcpy(q, d_A, sizeof(data_type) * n * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));

    free(h_work);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
}

void fill_random(float *mat, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mat[i * cols + j] = rand() % 100 / 100.0;
        }
    }
}

int main(int argc, char *argv[])
{
    constexpr int n = 16;
    constexpr float tolerance = 0.001;
    constexpr int max_iterations = 100;

    float *A = (float *)malloc(n * n * sizeof(float));
    fill_random(A, n, n);
    float *x = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    fill_random(b, n, 1);

    dr_bcg(A, n, x, b, tolerance, max_iterations);

    free(A);
    free(x);
    free(b);

    return 0;
}