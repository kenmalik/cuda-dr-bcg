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

    cusparseHandle_t cusparseH = NULL;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));

    const std::string matrix_file = argv[1];
    SuiteSparseMatrix ssm(matrix_file);

    int64_t *jc_d = nullptr;
    int64_t *ir_d = nullptr;
    float *vals_d = nullptr;

    float *x_d = nullptr;
    CUDA_CHECK(cudaMalloc(&x_d, sizeof(float) * ssm.rows()));

    float *b_d = nullptr;
    std::vector<float> b_h(ssm.rows(), 1);
    CUDA_CHECK(cudaMalloc(&b_d, sizeof(float) * b_h.size()));
    CUDA_CHECK(cudaMemcpy(b_d, b_h.data(), sizeof(float) * b_h.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&jc_d, sizeof(int64_t) * ssm.jc_size()));
    CUDA_CHECK(cudaMalloc(&ir_d, sizeof(int64_t) * ssm.ir_size()));
    CUDA_CHECK(cudaMalloc(&vals_d, sizeof(float) * ssm.nnz()));

    // Convert from default Matlab types
    std::vector<int64_t> jc_64i(ssm.jc_size());
    for (int i = 0; i < ssm.jc_size(); i++)
    {
        jc_64i[i] = static_cast<int64_t>(ssm.jc()[i]);
    }
    CUDA_CHECK(cudaMemcpy(jc_d, jc_64i.data(), sizeof(int64_t) * jc_64i.size(), cudaMemcpyHostToDevice));

    std::vector<int64_t> ir_64i(ssm.ir_size());
    for (int i = 0; i < ssm.ir_size(); i++)
    {
        ir_64i[i] = static_cast<int64_t>(ssm.ir()[i]);
    }
    CUDA_CHECK(cudaMemcpy(ir_d, ir_64i.data(), sizeof(int64_t) * ir_64i.size(), cudaMemcpyHostToDevice));

    std::vector<float> nonzeros_32f(ssm.nnz());
    for (int i = 0; i < ssm.nnz(); i++)
    {
        nonzeros_32f[i] = static_cast<float>(ssm.data()[i]);
    }
    CUDA_CHECK(cudaMemcpy(vals_d, nonzeros_32f.data(), sizeof(float) * nonzeros_32f.size(), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t A;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &A, ssm.rows(), ssm.cols(), ssm.nnz(),
        jc_d, ir_d, vals_d, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    const int n = ssm.rows();

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
    std::cerr << "TODO: Implement preconditioned DR-BCG" << std::endl;

    std::cout << iterations;

    return 0;
}
