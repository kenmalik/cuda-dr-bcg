#include "dr_bcg/device_buffer.h"
#include "dr_bcg/helper.h"

/**
 * @brief Constructor. Allocates all device buffers.
 * @param n n dimension
 * @param s s dimension
 */
DeviceBuffer::DeviceBuffer(int n, int s)
{
    allocate(n, s);
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
 * @param n n dimension
 * @param s s dimension
 */
void DeviceBuffer::allocate(int n, int s)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&w), sizeof(float) * n * s));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sigma), sizeof(float) * s * s));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&(this->s)), sizeof(float) * n * s));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&xi), sizeof(float) * s * s));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&zeta), sizeof(float) * s * s));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temp), sizeof(float) * n * s));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&residual), sizeof(float) * n));
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