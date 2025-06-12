#include <iostream>
#include <vector>

#include "dr_bcg/helper.h"
#include "dr_bcg/dr-bcg.h"

int main(int argc, char *argv[])
{
    constexpr int n = 16;
    constexpr float tolerance = 0.001;
    constexpr int max_iterations = 100;

    std::vector<float> A(n * n);
    fill_random(A.data(), n, n);
    std::vector<float> x(n);
    fill_random(x.data(), n, 1);
    std::vector<float> b(n);
    fill_random(b.data(), n, 1);

    dr_bcg::dr_bcg(A.data(), n, x.data(), b.data(), tolerance, max_iterations);

    return 0;
}