#pragma once

#include <optional>
#include <stdexcept>
#include <vector>
#include <random>

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

void fill_random(float *mat, const int rows, const int cols, const std::optional<int> seed = std::nullopt);

void fill_spd(float *mat, const int n, const std::optional<int> seed = std::nullopt);

void print_matrix(const float *mat, const int rows, const int cols);

void print_device_matrix(const float *d_mat, const int rows, const int cols);

void check_nan(const float *d_arr, size_t size, std::string step);

std::vector<double> read_matrix_bin(std::string filename);
