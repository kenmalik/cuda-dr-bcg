#include <iostream>

#include "dr_bcg/helper.h"
#include "dr_bcg/dr-bcg.h"

int main(int argc, char *argv[])
{
    constexpr int n = 16;
    constexpr float tolerance = 0.001;
    constexpr int max_iterations = 100;

    float *A = (float *)malloc(n * n * sizeof(float));
    fill_random(A, n, n);
    float *x = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    fill_random(b, n, 1);

    dr_bcg::dr_bcg(A, n, x, b, tolerance, max_iterations);

    free(A);
    free(x);
    free(b);

    return 0;
}