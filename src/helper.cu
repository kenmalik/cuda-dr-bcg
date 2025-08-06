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
void fill_random(float *mat, const int rows, const int cols, const std::optional<int> seed)
{
    int s;
    if (seed)
    {
        s = *seed;
    }
    else
    {
        std::random_device rd;
        s = rd();
    }
    std::mt19937 gen(s);
    std::uniform_real_distribution<> dist(0, 1);

    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            mat[j * rows + i] = dist(gen);
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
void fill_spd(float *mat, const int n, const std::optional<int> seed)
{
    fill_random(mat, n, n, seed);

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
    if (!input_file.is_open())
    {
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

/**
 * @brief CUDA kernel to copy upper triangular of a matrix stored in column-major order.
 *
 * @param dst Pointer to destination device matrix (n x n)
 * @param src Pointer to source device matrix (m x n)
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 */
__global__ void copy_upper_triangular_kernel(float *dst, float *src, const int m, const int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col <= row && row < n && col < n)
    {
        dst[row * n + col] = src[row * m + col];
    }
}

void copy_upper_triangular(float *dst, float *src, const int m, const int n)
{
    constexpr int block_n = 16;
    constexpr dim3 block_dim(block_n, block_n);
    dim3 grid_dim((n + block_n - 1) / block_n, (n + block_n - 1) / block_n);
    copy_upper_triangular_kernel<<<grid_dim, block_dim>>>(dst, src, m, n);
}

/**
 * @brief Computes the inverse of a matrix using LU factorization.
 *
 * @param cusolverH cuSOLVER handle
 * @param params cuSOLVER params
 * @param d_A Device pointer to the symmetric positive definite matrix to invert. Result is overwritten to pointed location.
 * @param n Matrix dimension
 */
void invert_square_matrix(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *d_A, const int n)
{
    // LU Decomposition
    size_t workspaceInBytesOnDevice = 0;
    void *d_work = nullptr;
    size_t workspaceInBytesOnHost = 0;
    void *h_work = nullptr;

    int info = 0;
    int *d_info = nullptr;

    std::vector<int64_t> h_Ipiv(n, 0);
    int64_t *d_Ipiv = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * h_Ipiv.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n,
                                               CUDA_R_32F, d_A, n, CUDA_R_32F,
                                               &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
    if (0 < workspaceInBytesOnHost)
    {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr)
        {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, n, n,
                                    CUDA_R_32F, d_A, n, d_Ipiv, CUDA_R_32F,
                                    d_work, workspaceInBytesOnDevice,
                                    h_work, workspaceInBytesOnHost, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    // Solve A * X = I for inverse
    std::vector<float> h_I(n * n, 0);
    float *d_I = nullptr;

    for (int i = 0; i < n; i++)
    {
        h_I.at(i * n + i) = 1;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_I), sizeof(float) * h_I.size()));
    CUDA_CHECK(cudaMemcpy(d_I, h_I.data(), sizeof(float) * h_I.size(), cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n,
                                    CUDA_R_32F, d_A, n, d_Ipiv, CUDA_R_32F,
                                    d_I, n, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaMemcpy(d_A, d_I, sizeof(float) * h_I.size(), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_I));

    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
}
