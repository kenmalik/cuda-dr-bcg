#include <iostream>
#include "dr_bcg/helper.h"

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
