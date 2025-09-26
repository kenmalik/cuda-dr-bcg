#include <tuple>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <filesystem>

#include <mat_utils/mat_reader.h>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

__global__ void set_val(float *A_d, float val, size_t num_elements)
{
    const int idx = blockIdx.x * blockDim.y + threadIdx.x;
    if (idx < num_elements)
    {
        A_d[idx] = val;
    }
}

class DeviceSuiteSparseMatrix
{
public:
    explicit DeviceSuiteSparseMatrix(mat_utils::MatReader &ssm_A)
    {
        CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(int64_t) * (ssm_A.rows() + 1)));
        CUDA_CHECK(cudaMalloc(&d_colInd, sizeof(int64_t) * ssm_A.nnz()));
        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(float) * ssm_A.nnz()));

        std::vector<size_t> rowCounts(ssm_A.rows(), 0);
        for (size_t j = 0; j < ssm_A.cols(); ++j)
        {
            for (size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1]; ++p)
            {
                ++rowCounts[ssm_A.ir()[p]];
            }
        }

        std::vector<size_t> csrRowPtr_sz(ssm_A.rows() + 1, 0);
        for (size_t i = 0; i < ssm_A.rows(); ++i)
            csrRowPtr_sz[i + 1] = csrRowPtr_sz[i] + rowCounts[i];

        std::vector<size_t> next = csrRowPtr_sz;

        std::vector<size_t> csrColInd_sz(ssm_A.nnz());
        std::vector<float> csrVal(ssm_A.nnz());

        for (size_t j = 0; j < ssm_A.cols(); ++j)
        {
            for (size_t p = ssm_A.jc()[j]; p < ssm_A.jc()[j + 1]; ++p)
            {
                size_t row = ssm_A.ir()[p];
                size_t dst = next[row]++;
                csrColInd_sz[dst] = j;
                csrVal[dst] = static_cast<float>(ssm_A.data()[p]);
            }
        }

        // Convert host indices to int64_t
        std::vector<int64_t> csrRowPtr64(ssm_A.rows() + 1);
        std::vector<int64_t> csrColInd64(ssm_A.nnz());
        for (size_t i = 0; i < csrRowPtr_sz.size(); ++i)
            csrRowPtr64[i] = static_cast<int64_t>(csrRowPtr_sz[i]);
        for (size_t k = 0; k < csrColInd_sz.size(); ++k)
            csrColInd64[k] = static_cast<int64_t>(csrColInd_sz[k]);

        CUDA_CHECK(cudaMemcpy(d_rowPtr, csrRowPtr64.data(),
                              sizeof(int64_t) * csrRowPtr64.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colInd, csrColInd64.data(),
                              sizeof(int64_t) * csrColInd64.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals, csrVal.data(),
                              sizeof(float) * csrVal.size(), cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &A_, ssm_A.rows(), ssm_A.cols(), ssm_A.nnz(),
            d_rowPtr, d_colInd, d_vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    ~DeviceSuiteSparseMatrix()
    {
        if (A_)
        {
            CUSPARSE_CHECK(cusparseDestroySpMat(A_));
        }
        if (d_rowPtr)
        {
            CUDA_CHECK(cudaFree(d_rowPtr));
            d_rowPtr = nullptr;
        }
        if (d_colInd)
        {
            CUDA_CHECK(cudaFree(d_colInd));
            d_colInd = nullptr;
        }
        if (d_vals)
        {
            CUDA_CHECK(cudaFree(d_vals));
            d_vals = nullptr;
        }
    }

    cusparseSpMatDescr_t &handle()
    {
        return A_;
    }

private:
    int64_t *d_rowPtr = nullptr;
    int64_t *d_colInd = nullptr;
    float *d_vals = nullptr;
    cusparseSpMatDescr_t A_{};
};

int main(int argc, char *argv[])
{
    int s;
    try
    {
        if (argc == 3)
        {
            s = 1;
        }
        else if (argc == 4)
        {
            s = std::atoi(argv[3]);
        }
        else
        {
            throw std::invalid_argument("Invalid arg count");
        }

        if (!std::filesystem::exists(argv[1]))
        {
            throw std::invalid_argument(std::string(argv[1]) + " does not exist.");
        }

        if (!std::filesystem::exists(argv[2]))
        {
            throw std::invalid_argument(std::string(argv[2]) + " does not exist.");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Usage: ./example_2 [spd matrix] [preconditioner] [block size]" << std::endl;
        std::cerr << std::endl
                  << "Error: " << e.what() << std::endl;
        return 1;
    }

    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    cusolverDnParams_t cusolverP;
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverP));

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate_v2(&cublasH));

    cusparseHandle_t cusparseH;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    const std::string A_file = argv[1];
    mat_utils::MatReader ssm_A(A_file, {"Problem"}, "A");
    DeviceSuiteSparseMatrix A{ssm_A};
    const int n = ssm_A.rows();

    const std::string L_file = argv[2];
    mat_utils::MatReader ssm_L(L_file, {}, "L");
    DeviceSuiteSparseMatrix L{ssm_L};

    cusparseDnMatDescr_t X;
    float *d_X = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * n * s));
    CUSPARSE_CHECK(cusparseCreateDnMat(&X, n, s, n, d_X, CUDA_R_32F, CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t B;
    float *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * n * s));

    constexpr int block_size = 256;
    const size_t num_elements = n * s;
    const size_t num_blocks = (num_elements + block_size - 1) / block_size;
    set_val<<<num_blocks, block_size>>>(d_B, 1, num_elements);

    CUSPARSE_CHECK(cusparseCreateDnMat(&B, n, s, n, d_B, CUDA_R_32F, CUSPARSE_ORDER_COL));

    constexpr float tolerance = std::numeric_limits<float>::epsilon();
    const int max_iterations = n;

    int iterations = 0;
    dr_bcg::dr_bcg(cusolverH, cusolverP, cublasH, cusparseH, A.handle(), X, B, L.handle(), tolerance, max_iterations, &iterations);

    std::cout << iterations << std::endl;

    CUSPARSE_CHECK(cusparseDestroyDnMat(X));
    CUSPARSE_CHECK(cusparseDestroyDnMat(B));

    CUSPARSE_CHECK(cusparseDestroy(cusparseH));
    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverP));

    return 0;
}
