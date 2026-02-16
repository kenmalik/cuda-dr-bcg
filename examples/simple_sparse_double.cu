#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "dr_bcg/helper.h"
#include "dr_bcg/sparse.h"

int main() {
    constexpr std::int64_t n = 100;
    constexpr std::int64_t s = 8;

    // A = tridiag(-1, 2, -1)
    std::vector<std::int64_t> row_offsets(n + 1);
    std::vector<std::int64_t> col_indices;
    std::vector<double> values;

    std::int64_t nnz = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        row_offsets[i] = nnz;

        if (i > 0) {
            col_indices.push_back(i - 1);
            values.push_back(-1.0);
            ++nnz;
        }

        col_indices.push_back(i);
        values.push_back(2.0);
        ++nnz;

        if (i < n - 1) {
            col_indices.push_back(i + 1);
            values.push_back(-1.0);
            ++nnz;
        }
    }
    row_offsets[n] = nnz;

    std::int64_t *d_row_offsets = nullptr;
    std::int64_t *d_col_indices = nullptr;
    double *d_values = nullptr;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, (n + 1) * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, nnz * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_offsets, row_offsets.data(),
                          (n + 1) * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, col_indices.data(),
                          nnz * sizeof(std::int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(double),
                          cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t A;
    CUSPARSE_CHECK(cusparseCreateCsr(&A, n, n, nnz, d_row_offsets,
                                     d_col_indices, d_values,
                                     CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // B = ones vector
    std::vector<double> h_B(n * s, 1.0);
    double *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * s * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), n * s * sizeof(double),
                          cudaMemcpyHostToDevice));

    cusparseDnMatDescr_t B;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&B, n, s, n, d_B, CUDA_R_64F, CUSPARSE_ORDER_COL));

    // X = zeros vector
    std::vector<double> h_X(n * s, 0.0);
    double *d_X = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, n * s * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), n * s * sizeof(double),
                          cudaMemcpyHostToDevice));

    cusparseDnMatDescr_t X;
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&X, n, s, n, d_X, CUDA_R_64F, CUSPARSE_ORDER_COL));

    // Solver parameters
    constexpr double tolerance = 1e-6;
    constexpr int max_iterations = 1000;

    // Optional callback to monitor convergence
    auto residual_callback = [](int iteration, double residual) {
        std::cout << iteration << ": " << residual << std::endl;
    };

    std::cout << "Solving system with DR-BCG..." << std::endl;

    int iterations =
        dr_bcg::dr_bcg(A, X, B, tolerance, max_iterations, residual_callback);

    std::cout << "\nConverged in " << iterations << " iterations" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, n * s * sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::cout << "\nSolution:" << std::endl;
    for (int i = 0; i < std::min(5, static_cast<int>(n)); ++i) {
        std::cout << "  X[" << i << "] = " << h_X[i] << "" << std::endl;
    }
    std::cout << "..." << std::endl;
    for (int i = std::max(0, static_cast<int>(n * s) - 5); i < n * s; ++i) {
        std::cout << "  X[" << i << "] = " << h_X[i] << "" << std::endl;
    }

    CUSPARSE_CHECK(cusparseDestroySpMat(A));
    CUSPARSE_CHECK(cusparseDestroyDnMat(B));
    CUSPARSE_CHECK(cusparseDestroyDnMat(X));

    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_X));

    return 0;
}
