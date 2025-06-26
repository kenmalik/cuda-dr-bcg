#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include "dr_bcg/helper.h"

/**
 * @brief Prints a matrix stored in column-major order.
 * 
 * @param mat Pointer to the matrix data (column-major)
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void print_matrix(const float *mat, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%7.2f ", mat[j * rows + i]);
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Prints a device matrix by copying it to host and calling print_matrix.
 * 
 * @param d_mat Device pointer to the matrix (column-major)
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void print_device_matrix(const float *d_mat, const int rows, const int cols) {
    std::vector<float> h_mat(rows * cols);
    CUDA_CHECK(cudaMemcpy(h_mat.data(), d_mat, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
    print_matrix(h_mat.data(), rows, cols);
}

/**
 * @brief Fills a matrix with random values in the range [0, 1).
 * 
 * @param mat Pointer to the matrix data (host)
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void fill_random(float *mat, const int rows, const int cols)
{
    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            mat[j * rows + i] = std::rand() % 100 / 100.0;
        }
    }
}

/**
 * @brief Fills a matrix with random values and makes it symmetric positive definite (SPD).
 * 
 * The matrix is filled with random values, then multiplied by its transpose to ensure SPD.
 * 
 * @param mat Pointer to the matrix data (host)
 * @param n Matrix dimensions
 */
void fill_spd(float *mat, const int n)
{
    fill_random(mat, n, n);

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    float alpha = 1.0 / n;
    float beta = 0.0;

    float *d_mat = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mat), sizeof(float) * n * n));
    CUDA_CHECK(cudaMemcpy(d_mat, mat, sizeof(float) * n * n, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasSgemm_v2(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
        &alpha, d_mat, n, d_mat, n, &beta, d_mat, n));

    CUDA_CHECK(cudaMemcpy(mat, d_mat, sizeof(float) * n * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_mat));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
}