#include <iostream>
#include <fstream>
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
void print_device_matrix(const float *d_mat, const int rows, const int cols)
{
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

/**
 * @brief Checks for NaN values in a device array and throws an exception if any are found.
 *
 * This function copies the contents of a device array to the host, checks for NaN values,
 * and throws a runtime exception if any NaN values are detected. The exception message
 * includes the step name provided as input.
 *
 * @param d_arr Device pointer to the array to check.
 * @param size Number of elements in the array.
 * @param step Description of the step after which the check is performed.
 *
 * @throws std::runtime_error if a NaN value is detected in the array.
 */
void check_nan(const float *d_arr, size_t size, std::string step)
{
    std::vector<float> h_arr(size);
    CUDA_CHECK(cudaMemcpy(h_arr.data(), d_arr, sizeof(float) * size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < size; ++i)
    {
        if (std::isnan(h_arr[i]))
        {
            throw std::runtime_error("NaN detected after step: " + step);
        }
    }
}

std::vector<double> read_matrix_bin(std::string filename)
{
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    input_file.seekg(0, std::ios::end);
    long long file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);

    size_t num_doubles = file_size / sizeof(double);
    std::vector<double> matrix(num_doubles);
    input_file.read(reinterpret_cast<char *>(matrix.data()), file_size);

    return matrix;
}
