#include <vector>

#include <gtest/gtest.h>

#include "dr_bcg/dr-bcg.h"
#include "dr_bcg/helper.h"

TEST(QuadraticForm, ScalarOutputCorrect)
{
    cublasHandle_t cublasH;

    std::vector<float> h_x = {1, 2, 3};
    std::vector<float> h_A = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};

    float *d_x;
    float *d_temp;
    float *d_A;

    float h_y;
    float *d_y;

    float alpha = 1.0;
    float beta = 0.0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(float) * h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), sizeof(float) * h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * h_x.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    dr_bcg::quadratic_form(cublasH, 3, 3, alpha, d_x, d_A, beta, d_temp, d_y);

    CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_y));

    ASSERT_EQ(h_y, 228);
}

TEST(QuadraticForm, MatrixOutputCorrect)
{
    constexpr int m = 3;
    constexpr int n = 2;

    cublasHandle_t cublasH;

    std::vector<float> h_x = {
        1, 3, 5,
        2, 4, 6};
    std::vector<float> h_A = {
        1, 4, 7,
        2, 5, 8,
        3, 6, 9};

    float *d_x;
    float *d_temp;
    float *d_A;

    std::vector<float> h_y(n * n);
    float *d_y;

    float alpha = 1.0;
    float beta = 0.0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(float) * h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(float) * h_y.size()));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * h_x.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    dr_bcg::quadratic_form(cublasH, m, n, alpha, d_x, d_A, beta, d_temp, d_y);

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, sizeof(float) * h_y.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_y));

    std::vector<float> expected = {
        549, 696,
        720, 912};

    ASSERT_EQ(h_y, expected);
}