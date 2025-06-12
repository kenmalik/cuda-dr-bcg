#include <iostream>
#include <vector>

#include "dr_bcg/helper.h"
#include "dr_bcg/dr-bcg.h"

int main(int argc, char *argv[])
{
    constexpr int n = 16;
    constexpr int m = 8;
    constexpr float tolerance = 0.001;
    constexpr int max_iterations = 100;

    std::vector<float> A(n * n);
    fill_random(A.data(), n, n);
    std::vector<float> X(n * m);
    fill_random(X.data(), n, m);
    std::vector<float> B(n * m);
    fill_random(B.data(), n, m);

    dr_bcg::dr_bcg(A.data(), n, m, X.data(), B.data(), tolerance, max_iterations);

    return 0;
}