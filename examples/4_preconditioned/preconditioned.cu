#include <tuple>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

#include <suitesparse_matrix.h>

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
    explicit DeviceSuiteSparseMatrix(SuiteSparseMatrix &ssm_A)
    {
        CUDA_CHECK(cudaMalloc(&d_jc_, sizeof(int64_t) * ssm_A.jc_size()));
        CUDA_CHECK(cudaMalloc(&d_ir_, sizeof(int64_t) * ssm_A.ir_size()));
        CUDA_CHECK(cudaMalloc(&d_vals_, sizeof(float) * ssm_A.nnz()));

        // Convert from default Matlab types
        std::vector<int64_t> jc_64i(ssm_A.jc_size());
        for (int i = 0; i < ssm_A.jc_size(); i++)
        {
            jc_64i[i] = static_cast<int64_t>(ssm_A.jc()[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_jc_, jc_64i.data(), sizeof(int64_t) * jc_64i.size(), cudaMemcpyHostToDevice));

        std::vector<int64_t> ir_64i(ssm_A.ir_size());
        for (int i = 0; i < ssm_A.ir_size(); i++)
        {
            ir_64i[i] = static_cast<int64_t>(ssm_A.ir()[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_ir_, ir_64i.data(), sizeof(int64_t) * ir_64i.size(), cudaMemcpyHostToDevice));

        std::vector<float> nonzeros_32f(ssm_A.nnz());
        for (int i = 0; i < ssm_A.nnz(); i++)
        {
            nonzeros_32f[i] = static_cast<float>(ssm_A.data()[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_vals_, nonzeros_32f.data(), sizeof(float) * nonzeros_32f.size(), cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &A_, ssm_A.rows(), ssm_A.cols(), ssm_A.nnz(),
            d_jc_, d_ir_, d_vals_, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    ~DeviceSuiteSparseMatrix()
    {
        if (A_)
        {
            CUSPARSE_CHECK(cusparseDestroySpMat(A_));
        }
        if (d_jc_)
        {
            CUDA_CHECK(cudaFree(d_jc_));
            d_jc_ = nullptr;
        }
        if (d_ir_)
        {
            CUDA_CHECK(cudaFree(d_ir_));
            d_ir_ = nullptr;
        }
        if (d_vals_)
        {
            CUDA_CHECK(cudaFree(d_vals_));
            d_vals_ = nullptr;
        }
    }

    cusparseSpMatDescr_t &handle()
    {
        return A_;
    }

private:
    int64_t *d_jc_ = nullptr;
    int64_t *d_ir_ = nullptr;
    float *d_vals_ = nullptr;
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
            s = std::atoi(argv[2]);
        }
        else
        {
            throw std::invalid_argument("Invalid arg count");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Usage: ./example_2 [spd matrix] [preconditioner] [block size]" << std::endl;
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
    SuiteSparseMatrix ssm_A(A_file);
    DeviceSuiteSparseMatrix A{ssm_A};

    const std::string L_file = argv[1];
    SuiteSparseMatrix ssm_L(L_file);
    DeviceSuiteSparseMatrix L{ssm_L};

    const int n = ssm_A.rows();

    cusparseDnMatDescr_t X;
    float *d_X = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, sizeof(float) * n * s));
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

    return 0;
}
