#include <iostream>

#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/layout/layout.h>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <cusolverDn.h>

#define CHECK_CUDA(call)                            \
    do                                              \
    {                                               \
        cudaError_t status = call;                  \
        if (status != CUDA_SUCCESS)                 \
        {                                           \
            std::cerr << "CUDA error" << std::endl; \
            exit(EXIT_FAILURE);                     \
        }                                           \
    } while (0)

#define CHECK_CUBLAS(call)                            \
    do                                                \
    {                                                 \
        cublasStatus_t status = call;                 \
        if (status != CUBLAS_STATUS_SUCCESS)          \
        {                                             \
            std::cerr << "cuBLAS error" << std::endl; \
            exit(EXIT_FAILURE);                       \
        }                                             \
    } while (0)

#define CHECK_CUSOLVER(call)                            \
    do                                                  \
    {                                                   \
        cusolverStatus_t status = call;                 \
        if (status != CUSOLVER_STATUS_SUCCESS)          \
        {                                               \
            std::cerr << "cuSolver error" << std::endl; \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

void print_matrix(const float *mat, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%6.3f ", mat[i * cols + j]);
        }
        std::cout << std::endl;
    }
}

/*
 * function [X_final, iterations] = DR_BCG(A, B, X, tol, maxit)
 *     iterations = 0;
 *     R = B - A * X;
 *     [w, sigma] = qr(R,'econ');
 *     s = w;

 *     for k = 1:maxit
 *         iterations = iterations + 1;
 *         xi = (s' * A * s)^-1;
 *         X = X + s * xi * sigma;
 *         if (norm(B(:,1) - A * X(:,1)) / norm(B(:,1))) < tol
 *             break
 *         else
 *             [w, zeta] = qr(w - A * s * xi,'econ');
 *             s = w + s * zeta';
 *             sigma = zeta * sigma;
 *         end
 *     end
 *     X_final = X;
 * end
 */
int dr_bcg(
    const float *A,
    const int n,
    const float *x,
    const float *b,
    const float tolerance,
    const int max_iterations)
{
    int iterations = 0;

    cublasHandle_t cuBLAS_handle;
    CHECK_CUBLAS(cublasCreate(&cuBLAS_handle));

    // r = b - Ax as GEMM:
    // r = -1.0 * Ax + r where r initially contains b
    const float alpha = -1;
    const float beta = 1;

    // Copy b into r
    float *r = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        r[i] = b[i];
    }

    CHECK_CUBLAS(cublasSgemv(
        cuBLAS_handle,
        CUBLAS_OP_N,
        n,
        n,
        &alpha,
        A, n,
        x, 1,
        &beta,
        r, 1));

    print_matrix(A, n, n);
    print_matrix(x, n, 1);
    print_matrix(r, n, 1);

    free(r);

    return 0;
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

int main(int argc, char *argv[])
{
    constexpr int n = 16;
    constexpr float tolerance = 0.001;
    constexpr int max_iterations = 100;

    float *A = (float *)malloc(n * n * sizeof(float));
    fill_random(A, n, n);
    float *x = (float *)malloc(n * sizeof(float));
    print_matrix(x, n, 1);
    float *b = (float *)malloc(n * sizeof(float));
    fill_random(b, n, 1);

    dr_bcg(A, n, x, b, tolerance, max_iterations);

    // // A is a column-major n*n matrix
    // cute::Layout A_layout = cute::make_layout(cute::make_shape(n, n), cute::LayoutLeft{});
    // cute::Tensor A = cute::make_tensor(A, A_layout);

    // cute::Layout vector_layout = cute::make_layout(cute::make_shape(n, 1), cute::LayoutLeft{});
    // cute::Tensor x = cute::make_tensor(x, vector_layout); // x is a column-major n*1 matrix
    // cute::Tensor b = cute::make_tensor(b, vector_layout); // b is a column-major n*1 matrix

    free(A);
    free(x);
    free(b);
}