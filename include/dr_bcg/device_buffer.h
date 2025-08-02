#pragma once

/**
 * @brief Device pointers for reused device buffers.
 *
 * This struct manages device memory for all buffers used in the DR-BCG algorithm.
 * It handles allocation and deallocation of all required device arrays.
 */
struct DeviceBuffer
{
    float *w = nullptr;        ///< Device pointer for matrix w (m x n)
    float *sigma = nullptr;    ///< Device pointer for matrix sigma (n x n)
    float *s = nullptr;        ///< Device pointer for matrix s (m x n)
    float *xi = nullptr;       ///< Device pointer for matrix xi (n x n)
    float *zeta = nullptr;     ///< Device pointer for matrix zeta (n x n)
    float *temp = nullptr;     ///< Device pointer for temporary matrix (m x n)
    float *residual = nullptr; ///< Device pointer for residual vector (m)

    DeviceBuffer(int m, int n);
    ~DeviceBuffer();

    void allocate(int m, int n);
    void deallocate();
};