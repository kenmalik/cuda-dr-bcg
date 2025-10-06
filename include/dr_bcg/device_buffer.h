#pragma once

/**
 * @brief Device pointers for reused device buffers.
 *
 * This struct manages device memory for all buffers used in the DR-BCG algorithm.
 * It handles allocation and deallocation of all required device arrays.
 */
struct DeviceBuffer
{
    float *w = nullptr;        ///< Device pointer for matrix w (n x s)
    float *sigma = nullptr;    ///< Device pointer for matrix sigma (s x s)
    float *s = nullptr;        ///< Device pointer for matrix s (n x s)
    float *xi = nullptr;       ///< Device pointer for matrix xi (s x s)
    float *zeta = nullptr;     ///< Device pointer for matrix zeta (s x s)
    float *temp = nullptr;     ///< Device pointer for temporary matrix (n x s)
    float *residual = nullptr; ///< Device pointer for residual vector (n)

    DeviceBuffer(int n, int s);
    ~DeviceBuffer();

    void allocate(int n, int s);
    void deallocate();
};