#pragma once

#define CUDA_CHECK(err)                                                            \
    do                                                                             \
    {                                                                              \
        cudaError_t err_ = (err);                                                  \
        if (err_ != cudaSuccess)                                                   \
        {                                                                          \
            fprintf(stderr, "CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                                \
        }                                                                          \
    } while (0)

#define CUSOLVER_CHECK(err)                                                            \
    do                                                                                 \
    {                                                                                  \
        cusolverStatus_t err_ = (err);                                                 \
        if (err_ != CUSOLVER_STATUS_SUCCESS)                                           \
        {                                                                              \
            fprintf(stderr, "cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusolver error");                                \
        }                                                                              \
    } while (0)

#define CUBLAS_CHECK(err)                                                            \
    do                                                                               \
    {                                                                                \
        cublasStatus_t err_ = (err);                                                 \
        if (err_ != CUBLAS_STATUS_SUCCESS)                                           \
        {                                                                            \
            fprintf(stderr, "cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                                \
        }                                                                            \
    } while (0)
