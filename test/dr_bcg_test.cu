#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/mismatch.h>

TEST(QuadraticForm, ScalarOutputCorrect) {
    cublasHandle_t cublasH;

    std::vector<float> h_x = {1, 2, 3};
    std::vector<float> h_A = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    float *d_x;
    float *d_temp;
    float *d_A;

    float h_y;
    float *d_y;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x),
                          sizeof(float) * h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp),
                          sizeof(float) * h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                          sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * h_x.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(),
                          cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    dr_bcg::quadratic_form(cublasH, 3, 3, d_x, d_A, d_temp, d_y);

    CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_y));

    ASSERT_EQ(h_y, 228);
}

TEST(QuadraticForm, MatrixOutputCorrect) {
    constexpr int m = 3;
    constexpr int n = 2;

    cublasHandle_t cublasH;

    std::vector<float> h_x = {1, 3, 5, 2, 4, 6};
    std::vector<float> h_A = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    float *d_x;
    float *d_temp;
    float *d_A;

    std::vector<float> h_y(n * n);
    float *d_y;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x),
                          sizeof(float) * h_x.size()));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_temp), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                          sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y),
                          sizeof(float) * h_y.size()));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * h_x.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(),
                          cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    dr_bcg::quadratic_form(cublasH, m, n, d_x, d_A, d_temp, d_y);

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, sizeof(float) * h_y.size(),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_y));

    std::vector<float> expected = {549, 696, 720, 912};

    ASSERT_EQ(h_y, expected);
}

TEST(Residual, OutputCorrect) {
    constexpr int m = 3;

    std::vector<float> h_B = {1, 2, 3, 2, 3, 4, 3, 4, 5};
    std::vector<float> h_X = {1, 2, 3, 2, 3, 4, 3, 4, 5};

    float *d_B = nullptr;
    float *d_X = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                          sizeof(float) * h_B.size()));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(float) * h_B.size(),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X),
                          sizeof(float) * h_X.size()));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), sizeof(float) * h_X.size(),
                          cudaMemcpyHostToDevice));

    std::vector<float> h_A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float *d_A;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                          sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(),
                          cudaMemcpyHostToDevice));

    std::vector<float> h_residual(m);
    float *d_residual = nullptr;

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_residual),
                          sizeof(float) * h_residual.size()));

    dr_bcg::residual(cublasH, d_residual, d_B, m, d_A, d_X);

    CUDA_CHECK(cudaMemcpy(h_residual.data(), d_residual,
                          sizeof(float) * h_residual.size(),
                          cudaMemcpyDeviceToHost));

    std::vector<float> expected = {-29, -34, -39};
    ASSERT_EQ(h_residual, expected);

    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_X));
}

TEST(InvertSquareMatrix, OutputCorrect) {
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
    for (int i = 0; i < m; i++) {
        I.at(i * m + i) = 1;
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * m));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A_inv), sizeof(float) * m * m));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_I), sizeof(float) * m * m));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_in.data(), sizeof(float) * h_A_in.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_inv, h_A_in.data(), sizeof(float) * h_A_in.size(),
                          cudaMemcpyHostToDevice));

    // Operation
    invert_square_matrix(cusolverH, cusolverParams, d_A, m);

    // Test A * A_inv = I
    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m,
                                &alpha, d_A, m, d_A_inv, m, &beta, d_I, m));

    CUDA_CHECK(cudaMemcpy(h_I_out.data(), d_I, sizeof(float) * h_I_out.size(),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_A_in.size(); i++) {
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

#ifdef DR_BCG_USE_THIN_QR

TEST(ThinQR, OutputCorrect) {
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

    for (auto &val : h_A_in) {
        val = dist(gen);
    }

    float *d_A = nullptr;
    float *d_Q = nullptr;
    float *d_R = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&d_Q, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(float) * m * m));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_in.data(), sizeof(float) * m * n,
                          cudaMemcpyHostToDevice));

    dr_bcg::thin_qr(cusolverH, params, cublasH, d_Q, d_R, m, n, d_A);

    std::cerr << "Q:" << std::endl;
    print_device_matrix(d_Q, m, n);
    std::cerr << "R:" << std::endl;
    print_device_matrix(d_R, n, n);

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                &alpha, d_Q, m, d_R, n, &beta, d_A, m));

    CUDA_CHECK(cudaMemcpy(h_A_out.data(), d_A, sizeof(float) * h_A_out.size(),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_R));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));

    std::cerr << "A in:" << std::endl;
    print_matrix(h_A_in.data(), m, n);
    std::cerr << "A out:" << std::endl;
    print_matrix(h_A_out.data(), m, n);

    for (int i = 0; i < h_A_in.size(); i++) {
        ASSERT_NEAR(h_A_in.at(i), h_A_out.at(i), test_tolerance);
    }
}

#else

TEST(QR_Factorization, OutputCorrect) {
    constexpr int m = 8;
    constexpr int n = 4;

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    thrust::device_vector<float> A(m * n);
    thrust::fill(A.begin(), A.end(), 1);
    float *d_A = thrust::raw_pointer_cast(A.data());

    thrust::device_vector<float> Q(m * n);
    float *d_Q = thrust::raw_pointer_cast(Q.data());

    thrust::device_vector<float> R(n * n);
    float *d_R = thrust::raw_pointer_cast(R.data());

    qr_factorization(cusolverH, params, d_Q, d_R, m, n, d_A);
    print_device_matrix(d_R, n, n);

    // Verification that A = Q * R
    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    thrust::device_vector<float> res(m * n);
    float *d_res = thrust::raw_pointer_cast(res.data());

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                                &alpha, d_Q, m, d_R, n, &beta, d_res, m));

    thrust::host_vector<float> h_A = A;
    thrust::host_vector<float> h_res = res;

    auto close_match = [](float a, float b) {
        constexpr float tolerance = 1e-6;
        return std::abs(a - b) <= tolerance;
    };
    auto [A_diff, res_diff] =
        thrust::mismatch(h_A.begin(), h_A.end(), h_res.begin(), close_match);

    std::cerr << "Mismatch at A: " << *A_diff << std::endl;
    std::cerr << "Mismatch at res: " << *res_diff << std::endl;

    ASSERT_EQ(A_diff, h_A.end());
    ASSERT_EQ(res_diff, h_res.end());
}

#endif // DR_BCG_USE_THIN_QR

// Copy upper triangular-like, even if source matrix may be taller than
// destination matrix
TEST(CopyUpperTriangular, OutputCorrect) {
    constexpr int m = 16;
    constexpr int n = 8;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0, 1);

    std::vector<float> A(m * n);
    std::generate(A.begin(), A.end(), [&]() { return distrib(gen); });

    std::vector<float> copy_expected(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) // Matrix is stored in column-major order
        {
            copy_expected.at(i * n + j) = A.at(i * m + j);
        }
    }

    std::vector<float> copy_got(copy_expected.size());

    float *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * A.size()));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * A.size(),
                          cudaMemcpyHostToDevice));

    float *d_copy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_copy, sizeof(float) * copy_got.size()));

    copy_upper_triangular(d_copy, d_A, m, n);

    CUDA_CHECK(cudaMemcpy(copy_got.data(), d_copy,
                          sizeof(float) * copy_got.size(),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_copy));

    std::cerr << "Original Matrix" << std::endl;
    print_matrix(A.data(), m, n);
    std::cerr << "Expected" << std::endl;
    print_matrix(copy_expected.data(), n, n);
    std::cerr << "Got" << std::endl;
    print_matrix(copy_got.data(), n, n);

    ASSERT_EQ(copy_expected, copy_got);
}

TEST(DR_BCG, OutputCorrect) {
    constexpr float check_tolerance = 0.01;

    constexpr int m = 32;
    constexpr int n = 8;
    constexpr float convergance_tolerance = 0.001;
    constexpr int max_iterations = 1000;

    std::vector<float> A(m * m);
    fill_spd(A.data(), m);
    std::vector<float> X(m * n, 0);
    std::vector<float> B_in(m * n);
    fill_random(B_in.data(), m, n);
    std::vector<float> B_out(m * n);

    // Operation
    auto [solution, iterations] =
        dr_bcg::dr_bcg(A, X, B_in, m, n, convergance_tolerance, max_iterations);

    // Test A * X = B
    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    float *d_A = nullptr;
    float *d_X = nullptr;
    float *d_B = nullptr;
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * A.size()));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X),
                          sizeof(float) * solution.size()));
    CUDA_CHECK(cudaMemcpy(d_X, solution.data(), sizeof(float) * solution.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                          sizeof(float) * B_out.size()));

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                                &alpha, d_A, m, d_X, m, &beta, d_B, m));
    CUDA_CHECK(cudaMemcpy(B_out.data(), d_B, sizeof(float) * B_out.size(),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < B_in.size(); i++) {
        float diff = std::abs(B_out.at(i) - B_in.at(i));
        ASSERT_LT(diff, check_tolerance);
    }

    // Free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_X));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
}

TEST(SPTRI_LeftMultiply, IdentityStaysSame) {
    cusparseHandle_t cusparseH;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    constexpr int m = 8;
    constexpr int n = 4;

    constexpr cudaDataType_t data_type = CUDA_R_32F;
    constexpr cusparseOrder_t order = CUSPARSE_ORDER_COL;

    constexpr cusparseIndexType_t index_type = CUSPARSE_INDEX_64I;
    constexpr cusparseIndexBase_t base_type = CUSPARSE_INDEX_BASE_ZERO;

    // A = I
    thrust::device_vector<float> A_vals(m);
    thrust::fill(A_vals.begin(), A_vals.end(), 1);

    auto counter = thrust::make_counting_iterator<int64_t>(0);

    thrust::device_vector<int64_t> A_row_offsets(m + 1);
    thrust::copy_n(counter, A_row_offsets.size(), A_row_offsets.begin());
    int64_t *row_offsets = thrust::raw_pointer_cast(A_row_offsets.data());

    counter = thrust::make_counting_iterator<int64_t>(0);
    thrust::device_vector<int64_t> A_col_indices(m);
    thrust::copy_n(counter, A_col_indices.size(), A_col_indices.begin());
    int64_t *col_indices = thrust::raw_pointer_cast(A_col_indices.data());

    thrust::device_vector<float> values(m * n);
    thrust::fill(values.begin(), values.end(), 1);
    float *d_values = thrust::raw_pointer_cast(values.data());

    cusparseSpMatDescr_t A_desc;
    CUSPARSE_CHECK(cusparseCreateCsr(&A_desc, m, m, m, row_offsets, col_indices,
                                     d_values, index_type, index_type,
                                     base_type, data_type));

    // B = 1
    thrust::device_vector<float> B(m * n);
    thrust::fill(B.begin(), B.end(), 1);
    float *d_B = thrust::raw_pointer_cast(B.data());
    cusparseDnMatDescr_t B_desc;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&B_desc, m, n, m, d_B, data_type, order));

    // C initialized
    thrust::device_vector<float> C(m * n);
    float *d_C = thrust::raw_pointer_cast(C.data());
    cusparseDnMatDescr_t C_desc;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&C_desc, m, n, m, d_C, data_type, order));

    constexpr cusparseOperation_t op_type = CUSPARSE_OPERATION_NON_TRANSPOSE;
    sptri_left_multiply(cusparseH, C_desc, op_type, A_desc, B_desc);

    // Verify C == B
    thrust::host_vector<float> expected = B;
    thrust::host_vector<float> got = C;

    auto close_match = [](float a, float b) {
        constexpr float tolerance = 1e-6;
        return std::abs(a - b) <= tolerance;
    };
    auto [expected_diff, got_diff] =
        thrust::mismatch(expected.begin(), expected.end(), got.begin(), close_match);

    std::cerr << "Mismatch at A: " << *expected_diff << std::endl;
    std::cerr << "Mismatch at res: " << *got_diff << std::endl;

    ASSERT_EQ(expected_diff, expected.end());
    ASSERT_EQ(got_diff, got.end());

    CUSPARSE_CHECK(cusparseDestroySpMat(A_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(B_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(C_desc));

    CUSPARSE_CHECK(cusparseDestroy(cusparseH));
}