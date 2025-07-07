#include <vector>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "dr_bcg/dr_bcg.h"
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

TEST(QR_Factorization, OutputCorrect)
{
    constexpr float tolerance = 0.001;

    constexpr int m = 8;
    constexpr int n = 4;

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t cusolverParams;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

    float *d_A = nullptr;
    float *d_Q = nullptr;
    float *d_R = nullptr;

    std::vector<float> h_A_in(m * n);
    fill_random(h_A_in.data(), m, n);
    std::vector<float> h_A_out(m * n);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(float) * n * n));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_in.data(), sizeof(float) * h_A_in.size(), cudaMemcpyHostToDevice));

    // Operation
    dr_bcg::qr_factorization(cusolverH, cusolverParams, d_Q, d_R, m, n, d_A);

    // Test A = Q * R
    CUDA_CHECK(cudaMemset(d_A, 0, sizeof(float) * m * n));

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                &alpha, d_Q, m, d_R, n,
                                &beta, d_A, m));

    CUDA_CHECK(cudaMemcpy(h_A_out.data(), d_A, sizeof(float) * h_A_out.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_A_in.size(); i++)
    {
        float diff = std::abs(h_A_in.at(i) - h_A_out.at(i));
        ASSERT_LT(diff, tolerance);
    }

    // Free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_R));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverParams));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
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

    float *d_B = nullptr;
    float *d_X = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * h_B.size()));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(float) * h_B.size(), cudaMemcpyHostToDevice));

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

    dr_bcg::residual(cublasH, d_residual, d_B, m, d_A, d_X);

    CUDA_CHECK(cudaMemcpy(h_residual.data(), d_residual, sizeof(float) * h_residual.size(), cudaMemcpyDeviceToHost));

    std::vector<float> expected = {-29, -34, -39};
    ASSERT_EQ(h_residual, expected);

    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_X));
}

TEST(InvertSquareMatrix, OutputCorrect)
{
    constexpr float tolerance = 0.001;

    constexpr int m = 8;

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t cusolverParams;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

    float *d_A = nullptr;
    float *d_A_inv = nullptr;
    float *d_I = nullptr;

    std::vector<float> h_A_in(m * m);
    fill_spd(h_A_in.data(), m);
    std::vector<float> h_I_out(m * m);

    std::vector<float> I(m * m, 0);
    for (int i = 0; i < m; i++)
    {
        I.at(i * m + i) = 1;
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_inv), sizeof(float) * m * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_I), sizeof(float) * m * m));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_in.data(), sizeof(float) * h_A_in.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_inv, h_A_in.data(), sizeof(float) * h_A_in.size(), cudaMemcpyHostToDevice));

    // Operation
    dr_bcg::invert_square_matrix(cusolverH, cusolverParams, d_A, m);

    // Test A * A_inv = I
    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m,
                                &alpha, d_A, m, d_A_inv, m,
                                &beta, d_I, m));

    CUDA_CHECK(cudaMemcpy(h_I_out.data(), d_I, sizeof(float) * h_I_out.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_A_in.size(); i++)
    {
        float diff = std::abs(I.at(i) - h_I_out.at(i));
        ASSERT_LT(diff, tolerance);
    }

    // Free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_A_inv));
    CUDA_CHECK(cudaFree(d_I));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverParams));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
}

TEST(ThinQR, OutputCorrect)
{
    constexpr float test_tolerance = 1e-6;

    constexpr int m = 32;
    constexpr int n = 4;

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    std::vector<float> h_A_in(m * n);
    std::vector<float> h_A_out(m * n);

    for (auto &val : h_A_in)
    {
        val = dist(gen);
    }

    float *d_A = nullptr;
    float *d_Q = nullptr;
    float *d_R = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&d_Q, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(float) * m * m));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_in.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));

    dr_bcg::thin_qr(cusolverH, params, cublasH, d_Q, d_R, m, n, d_A);

    std::cout << "Q:" << std::endl;
    print_device_matrix(d_Q, m, n);
    std::cout << "R:" << std::endl;
    print_device_matrix(d_R, n, n);

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
        &alpha, d_Q, m, d_R, n,
        &beta, d_A, m));

    CUDA_CHECK(cudaMemcpy(h_A_out.data(), d_A, sizeof(float) * h_A_out.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_R));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));

    std::cout << "A in:" << std::endl;
    print_matrix(h_A_in.data(), m, n);
    std::cout << "A out:" << std::endl;
    print_matrix(h_A_out.data(), m, n);

    for (int i = 0; i < h_A_in.size(); i++)
    {
        ASSERT_NEAR(h_A_in.at(i), h_A_out.at(i), test_tolerance);
    }
}

TEST(DR_BCG, OutputCorrect)
{
    constexpr float check_tolerance = 0.01;

    constexpr int m = 32;
    constexpr int n = 8;
    constexpr float convergance_tolerance = 0.001;
    constexpr int max_iterations = 100;

    std::vector<float> A(m * m);
    fill_spd(A.data(), m);
    std::vector<float> X(m * n, 0);
    std::vector<float> B_in(m * n);
    fill_random(B_in.data(), m, n);
    std::vector<float> B_out(m * n);

    // Operation
    auto [solution, iterations] = dr_bcg::dr_bcg(A, X, B_in, m, n, convergance_tolerance, max_iterations);

    // Test A * X = B
    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    float *d_A = nullptr;
    float *d_X = nullptr;
    float *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * A.size()));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * solution.size()));
    CUDA_CHECK(cudaMemcpy(d_X, solution.data(), sizeof(float) * solution.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * B_out.size()));

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                                &alpha, d_A, m, d_X, m,
                                &beta, d_B, m));
    CUDA_CHECK(cudaMemcpy(B_out.data(), d_B, sizeof(float) * B_out.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < B_in.size(); i++)
    {
        float diff = std::abs(B_out.at(i) - B_in.at(i));
        ASSERT_LT(diff, check_tolerance);
    }

    // Free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_X));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
}