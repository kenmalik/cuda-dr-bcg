#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <nvtx3/nvtx3.hpp>

#include "dr_bcg/helper.h"

/**
 * @brief Checks CUDA runtime API results and throws an exception on error.
 *
 * @param err CUDA error code returned by a CUDA runtime API call.
 * @param func Name of the function where the check is performed.
 * @param file Source file name where the check is performed.
 * @param line Line number in the source file where the check is performed.
 *
 * @throws std::runtime_error if the CUDA error code is not cudaSuccess.
 */
void check(cudaError_t err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::ostringstream oss;
        oss << "CUDA error " << err
            << " at " << file << " line " << line
            << ": " << func << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Checks cuSOLVER API results and throws an exception on error.
 *
 * @param err CUDA error code returned by a CUDA runtime API call.
 * @param func Name of the function where the check is performed.
 * @param file Source file name where the check is performed.
 * @param line Line number in the source file where the check is performed.
 *
 * @throws std::runtime_error if the cuSOLVER error code is not CUSOLVER_STATUS_SUCCESS.
 */
void check(cusolverStatus_t err, const char *const func, const char *const file, const int line)
{
    if (err != CUSOLVER_STATUS_SUCCESS)
    {
        std::ostringstream oss;
        oss << "cuSOLVER error " << err
            << " at " << file << " line " << line
            << ": " << func;
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Checks cuBLAS API results and throws an exception on error.
 *
 * @param err cuBLAS error code returned by a cuBLAS API call.
 * @param func Name of the function where the check is performed.
 * @param file Source file name where the check is performed.
 * @param line Line number in the source file where the check is performed.
 *
 * @throws std::runtime_error if the cuBLAS error code is not CUBLAS_STATUS_SUCCESS.
 */
void check(cublasStatus_t err, const char *const func, const char *const file, const int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::ostringstream oss;
        oss << "cuBLAS error " << err
            << " at " << file << " line " << line
            << ": " << func;
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Checks cuSPARSE API results and throws an exception on error.
 *
 * @param err cuSPARSE error code returned by a cuSPARSE API call.
 * @param func Name of the function where the check is performed.
 * @param file Source file name where the check is performed.
 * @param line Line number in the source file where the check is performed.
 *
 * @throws std::runtime_error if the cuSPARSE error code is not CUSPARSE_STATUS_SUCCESS.
 */
void check(cusparseStatus_t err, const char *const func, const char *const file, const int line)
{
    if (err != CUSPARSE_STATUS_SUCCESS)
    {
        std::ostringstream oss;
        oss << "cuSPARSE error " << err
            << " at " << file << " line " << line
            << ": " << func << ": " << cusparseGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

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

#ifdef USE_THIN_QR

void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A)
{
    throw std::runtime_error("qr_factorization not built");
}

#else

/**
 * @brief Computes the QR factorization of matrix A.
 *
 * @param cusolverH cuSOLVER handle
 * @param params Params for the cuSOLVER handle
 * @param Q Pointer to device memory to store Q result in
 * @param R Pointer to device memory to store R result in. Note that the lower triangular still contains householder vectors and must be handled accordingly
 * (e.g. by using trmm in future multiplications using the R factor)
 * @param m m-dimension (leading dimension) of A
 * @param n n-dimension (second dimension) of A
 * @param A The matrix to factorize (device pointer)
 */
void qr_factorization(cusolverDnHandle_t &cusolverH, cusolverDnParams_t &params, float *Q, float *R, const int m, const int n, const float *A)
{
    NVTX3_FUNC_RANGE();

    int k = std::min(m, n);
    int info = 0;

    float *d_tau = nullptr;
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(float) * k));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    void *d_work = nullptr;
    size_t lwork_geqrf_h = 0;
    void *h_work = nullptr;

    CUDA_CHECK(cudaMemcpy(Q, A, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

    // Create device buffer
    size_t lwork_geqrf_bytes_d = 0;
    CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(cusolverH, params, m, n, CUDA_R_32F, Q,
                                               m, CUDA_R_32F, d_tau,
                                               CUDA_R_32F, &lwork_geqrf_bytes_d,
                                               &lwork_geqrf_h));

    int lwork_orgqr = 0;
    CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolverH, m, n, k, Q, m, d_tau, &lwork_orgqr));

    // Note: The legacy cuSOLVER API returns lwork number of array values
    // while the generic API returns lwork in bytes.
    // This is why we multiply lwork_orgqr by sizeof(float) to get a
    // proper comparison in workspace sizes.
    const size_t lwork_bytes_d = std::max(lwork_geqrf_bytes_d, lwork_orgqr * sizeof(float));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), lwork_bytes_d));

    if (lwork_geqrf_h > 0)
    {
        h_work = reinterpret_cast<void *>(malloc(lwork_geqrf_h));
        if (h_work == nullptr)
        {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    CUSOLVER_CHECK(cusolverDnXgeqrf(cusolverH, params, m, n, CUDA_R_32F, Q,
                                    m, CUDA_R_32F, d_tau,
                                    CUDA_R_32F, d_work, lwork_bytes_d, h_work,
                                    lwork_geqrf_h, d_info));
    if (h_work)
    {
        free(h_work); // No longer needed
    }

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }

    const int max_R_col = std::min(m, n);
    copy_upper_triangular(R, Q, m, n);

    // Explicitly compute Q
    CUSOLVER_CHECK(cusolverDnSorgqr(cusolverH, m, n, k, Q, m, d_tau, reinterpret_cast<float *>(d_work), lwork_bytes_d, d_info));
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }

    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
}

#endif

#ifndef USE_THIN_QR

void thin_qr(
    cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &params,
    cublasHandle_t &cublasH,
    float *Q,
    float *R,
    const int m,
    const int n,
    const float *A)
{
    throw std::runtime_error("thin_qr not built");
}

#else

void thin_qr(
    cusolverDnHandle_t &cusolverH,
    cusolverDnParams_t &params,
    cublasHandle_t &cublasH,
    float *Q,
    float *R,
    const int m,
    const int n,
    const float *A)
{
    NVTX3_FUNC_RANGE();

    // H = M^T * M
    float *d_H = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_H), sizeof(float) * n * n));

    constexpr float alpha = 1;
    constexpr float beta = 0;
    CUBLAS_CHECK(cublasSgemm_v2(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
        &alpha, A, m, A, m,
        &beta, d_H, n));

    // R^T * R = H
    void *h_work = nullptr;
    size_t h_lwork_Xpotrf = 0;
    void *d_work = nullptr;
    size_t d_lwork_Xpotrf = 0;

    int info = 0;
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
        cusolverH, params, CUBLAS_FILL_MODE_UPPER, n,
        CUDA_R_32F, d_H, n,
        CUDA_R_32F, &d_lwork_Xpotrf, &h_lwork_Xpotrf));

    if (h_lwork_Xpotrf > 0)
    {
        h_work = reinterpret_cast<void *>(malloc(h_lwork_Xpotrf));
        if (h_work == nullptr)
        {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_lwork_Xpotrf));

    CUSOLVER_CHECK(cusolverDnXpotrf(
        cusolverH, params, CUBLAS_FILL_MODE_UPPER, n,
        CUDA_R_32F, d_H, n,
        CUDA_R_32F, d_work, d_lwork_Xpotrf, h_work, h_lwork_Xpotrf, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }
    if (info > 0)
    {
        throw std::runtime_error("cusolverDnXpotrf (Cholesky factorization) failed. The smallest leading minor of d_H which is not positive definite is " + std::to_string(info));
    }

    copy_upper_triangular(R, d_H, n, n);

    // Q = M * R^-1
    size_t d_lwork_Xtrtri = 0;
    size_t h_lwork_Xtrtri = 0;
    info = 0;

    CUSOLVER_CHECK(cusolverDnXtrtri_bufferSize(
        cusolverH, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, n,
        CUDA_R_32F, d_H, n,
        &d_lwork_Xtrtri, &h_lwork_Xtrtri));

    if (h_lwork_Xtrtri > h_lwork_Xpotrf)
    {
        if (h_work != nullptr)
        {
            free(h_work);
        }
        h_work = reinterpret_cast<void *>(malloc(h_lwork_Xtrtri));
    }
    if (d_lwork_Xtrtri > d_lwork_Xpotrf)
    {
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_lwork_Xtrtri));
    }

    CUSOLVER_CHECK(cusolverDnXtrtri(
        cusolverH, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, n,
        CUDA_R_32F, d_H, n,
        d_work, d_lwork_Xtrtri, h_work, h_lwork_Xtrtri, d_info));

    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0)
    {
        throw std::runtime_error(std::to_string(-info) + "-th parameter is wrong \n");
    }

    CUBLAS_CHECK(cublasStrmm_v2(
        cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
        &alpha, d_H, n, A, m, Q, m));

    if (h_work != nullptr)
    {
        free(h_work);
    }
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));

    CUDA_CHECK(cudaFree(d_H));
}

#endif

void print_sparse_matrix(const cusparseHandle_t &cusparseH, const cusparseSpMatDescr_t &sp_mat)
{
    constexpr cusparseSparseToDenseAlg_t ALG = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;

    size_t buffer_size = 0;
    void *buffer = nullptr;

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;

    float *dense_d = nullptr;
    cusparseDnMatDescr_t dense{};

    CUSPARSE_CHECK(cusparseSpMatGetSize(sp_mat, &rows, &cols, &nnz));
    CUDA_CHECK(cudaMalloc(&dense_d, sizeof(float) * rows * cols));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense, rows, cols, rows, dense_d, CUDA_R_32F, CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseSparseToDense_bufferSize(cusparseH, sp_mat, dense, ALG, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    CUSPARSE_CHECK(cusparseSparseToDense(cusparseH, sp_mat, dense, ALG, buffer));

    print_device_matrix(dense_d, rows, cols);

    CUDA_CHECK(cudaFree(buffer));
    CUDA_CHECK(cudaFree(dense_d));
    CUSPARSE_CHECK(cusparseDestroyDnMat(dense));
}

void sptri_left_multiply(
    const cusparseHandle_t &cusparseH,
    cusparseDnMatDescr_t &C,
    cusparseOperation_t opA,
    const cusparseSpMatDescr_t &A,
    const cusparseDnMatDescr_t &B)
{
    constexpr cusparseOperation_t OP_B = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr float alpha = 1;
    constexpr cudaDataType COMPUTE_TYPE = CUDA_R_32F;
    constexpr cusparseSpSMAlg_t ALG_TYPE = CUSPARSE_SPSM_ALG_DEFAULT;

    cusparseSpSMDescr_t spsm{};
    CUSPARSE_CHECK(cusparseSpSM_createDescr(&spsm));

    void *buffer = nullptr;
    size_t buffer_size = 0;

    CUSPARSE_CHECK(cusparseSpSM_bufferSize(cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha),
                                           A, B, C, COMPUTE_TYPE, ALG_TYPE, spsm, &buffer_size));

    if (buffer_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    }
    else
    {
        throw std::runtime_error("s solve: buffer not allocated");
    }

    CUSPARSE_CHECK(cusparseSpSM_analysis(cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha),
                                         A, B, C, COMPUTE_TYPE, ALG_TYPE, spsm, buffer));

    CUSPARSE_CHECK(cusparseSpSM_solve(cusparseH, opA, OP_B, reinterpret_cast<const void *>(&alpha),
                                      A, B, C, COMPUTE_TYPE, ALG_TYPE, spsm));

    CUDA_CHECK(cudaFree(buffer));
    CUSPARSE_CHECK(cusparseSpSM_destroyDescr(spsm));
}
