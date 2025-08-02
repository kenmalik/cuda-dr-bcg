#include "dr_bcg/device_buffer.h"
#include "dr_bcg/helper.h"

/**
 * @brief Constructor. Allocates all device buffers.
 * @param m m dimension
 * @param n n dimension
 */
DeviceBuffer::DeviceBuffer(int m, int n)
{
    allocate(m, n);
}

/**
 * @brief Destructor. Frees all allocated device memory.
 */
DeviceBuffer::~DeviceBuffer()
{
    deallocate();
}

/**
 * @brief Allocates device memory for all buffers.
 * @param m m dimension
 * @param n n dimension
 */
void DeviceBuffer::allocate(int m, int n)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&w), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sigma), sizeof(float) * n * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&xi), sizeof(float) * n * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&zeta), sizeof(float) * n * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temp), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&residual), sizeof(float) * m));
}

/**
 * @brief Deallocates all device memory.
 */
void DeviceBuffer::deallocate()
{
    CUDA_CHECK(cudaFree(w));
    CUDA_CHECK(cudaFree(sigma));
    CUDA_CHECK(cudaFree(s));
    CUDA_CHECK(cudaFree(xi));
    CUDA_CHECK(cudaFree(zeta));
    CUDA_CHECK(cudaFree(temp));
    CUDA_CHECK(cudaFree(residual));
}