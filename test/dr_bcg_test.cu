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

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(float) * h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), sizeof(float) * h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * h_x.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    dr_bcg::quadratic_form(cublasH, 3, 3, d_x, d_A, d_temp, d_y);

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

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(float) * h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(float) * h_y.size()));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * h_x.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    dr_bcg::quadratic_form(cublasH, m, n, d_x, d_A, d_temp, d_y);

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

TEST(Residual, OutputCorrect)
{
    constexpr int m = 3;

    std::vector<float> h_B = {
        1, 2, 3,
        2, 3, 4,
        3, 4, 5};
    std::vector<float> h_X = {
        1, 2, 3,
        2, 3, 4,
        3, 4, 5};
    float *d_X;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * h_X.size()));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), sizeof(float) * h_X.size(), cudaMemcpyHostToDevice));

    std::vector<float> h_A = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    float *d_A;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(), cudaMemcpyHostToDevice));

    std::vector<float> h_residual(m);
    float *d_residual = nullptr;

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_residual), sizeof(float) * h_residual.size()));

    dr_bcg::residual(cublasH, d_residual, h_B.data(), m, d_A, d_X);

    CUDA_CHECK(cudaMemcpy(h_residual.data(), d_residual, sizeof(float) * h_residual.size(), cudaMemcpyDeviceToHost));

    std::vector<float> expected = {-29, -34, -39};
    ASSERT_EQ(h_residual, expected);
}
