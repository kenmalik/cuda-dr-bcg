#include <iostream>
#include <cublas_v2.h>
#include "dr_bcg/helper.h"

/// @brief Prints a matrix stored in column-major order
void print_matrix(const float *mat, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%6.3f ", mat[j * cols + i]);
        }
        std::cout << std::endl;
    }
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

void fill_spd(float *mat, const int n) {
    fill_random(mat, n, n);
    
    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    float alpha = 1.0;
    float beta = 0.0;

    float *d_mat = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mat), sizeof(float) * n * n));
    CUDA_CHECK(cudaMemcpy(d_mat, mat, sizeof(float) * n * n, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasSgemm_v2(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
        &alpha, d_mat, n, d_mat, n, &beta, d_mat, n
    ));

    CUDA_CHECK(cudaMemcpy(mat, d_mat, sizeof(float) * n * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_mat));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
}