#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

#include <cuda/std/cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/mismatch.h>

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
bool match(const thrust::host_vector<T> &a, const thrust::host_vector<T> &b,
           T tolerance = 1e-6) {
    if (a.size() != b.size()) {
        return false;
    }

    auto float_compare = [tolerance](T x, T y) {
        return cuda::std::abs(x - y) <= tolerance;
    };
    auto [a_diff, b_diff] =
        thrust::mismatch(a.begin(), a.end(), b.begin(), float_compare);

    return a_diff == a.end() && b_diff == b.end();
}

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

TEST(QuadraticForm, SparseIdentityCheck) {
    constexpr int m = 8;
    constexpr int n = 4;

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    cusparseHandle_t cusparseH;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    // A = I
    constexpr cudaDataType_t data_type = CUDA_R_32F;
    constexpr cusparseOrder_t order = CUSPARSE_ORDER_COL;
    constexpr cusparseIndexType_t index_type = CUSPARSE_INDEX_64I;
    constexpr cusparseIndexBase_t base_type = CUSPARSE_INDEX_BASE_ZERO;

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

    // X = 1
    thrust::device_vector<float> X(m * n);
    thrust::fill(X.begin(), X.end(), 1);
    float *d_X = thrust::raw_pointer_cast(X.data());
    cusparseDnMatDescr_t X_desc;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&X_desc, m, n, m, d_X, data_type, order));

    float *d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * m * n));

    thrust::device_vector<float> Y(n * n);
    float *d_Y = thrust::raw_pointer_cast(Y.data());

    dr_bcg::quadratic_form(cublasH, cusparseH, m, n, X_desc, A_desc, d_work,
                           d_Y);

    CUDA_CHECK(cudaFree(d_work));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    // Expected = 1_{n * n} * m
    thrust::host_vector<float> got = Y;
    thrust::host_vector<float> expected(n * n);
    thrust::fill(expected.begin(), expected.end(), m);
    ASSERT_TRUE(match(expected, got));
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

TEST(InvertSquareMatrix, TwoByTwoMatrix) {
    constexpr int m = 2;

    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    std::vector<float> vals = {1, 3, 2, 4};
    thrust::device_vector<float> A(vals.begin(), vals.end());
    float *d_A = thrust::raw_pointer_cast(A.data());

    invert_square_matrix(cusolverH, params, d_A, m);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));

    std::vector<float> expected_vals = {-2, 1.5f, 1, -0.5f};
    thrust::host_vector<float> expected(expected_vals.begin(),
                                        expected_vals.end());
    std::cerr << "expected" << std::endl;
    print_matrix(thrust::raw_pointer_cast(expected.data()), m, m);

    thrust::host_vector<float> got = A;
    print_device_matrix(d_A, 2, 2);
    ASSERT_TRUE(match(expected, got));
}

TEST(InvertSquareMatrix, DiagonalMatrix) {
    constexpr int m = 8;

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t cusolverParams;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

    struct diagonal_functor {
        const int N;
        const float X;
        diagonal_functor(int n, float x) : N(n), X(x) {}

        __host__ __device__ float operator()(int index) const {
            int row = index / N;
            int col = index % N;
            return (row == col) ? X : 0;
        }
    };

    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end = begin + m * m;

    constexpr float fill_value = 10;
    thrust::host_vector<float> I(m * m);
    thrust::transform(begin, end, I.begin(), diagonal_functor(m, fill_value));

    thrust::device_vector<float> A = I;
    float *d_A = thrust::raw_pointer_cast(A.data());

    // Operation
    invert_square_matrix(cusolverH, cusolverParams, d_A, m);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverParams));

    // Test that result contains reciprocal of fill value on diagonal
    thrust::host_vector<float> expected(m * m);
    thrust::transform(begin, end, expected.begin(),
                      diagonal_functor(m, 1 / fill_value));
    thrust::host_vector<float> got = A;
    ASSERT_TRUE(match(expected, got));
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

TEST(QR_Factorization, ProductOfFactorsIsA) {
    constexpr int m = 8;
    constexpr int n = 4;

    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    thrust::device_vector<float> A(m * n);
    thrust::fill(A.begin(), A.end(), 2);
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

    thrust::host_vector<float> expected = A;
    thrust::host_vector<float> got = res;
    ASSERT_TRUE(match(expected, got));
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

    CUSPARSE_CHECK(cusparseDestroySpMat(A_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(B_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(C_desc));

    CUSPARSE_CHECK(cusparseDestroy(cusparseH));

    thrust::host_vector<float> expected = B;
    thrust::host_vector<float> got = C;
    ASSERT_TRUE(match(expected, got));
}