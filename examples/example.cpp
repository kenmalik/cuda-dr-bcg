#include <iostream>
#include <vector>

#include "dr_bcg/helper.h"
#include "dr_bcg/dr-bcg.h"

int main(int argc, char *argv[])
{
    constexpr int m = 16;
    constexpr int n = 8;
    constexpr float tolerance = 0.001;
    constexpr int max_iterations = 25;

    std::vector<float> A(m * m);
    fill_random(A.data(), m, m);
    std::vector<float> X(m * n);
    fill_random(X.data(), m, n);
    std::vector<float> B(m * n);
    fill_random(B.data(), m, n);

    dr_bcg::dr_bcg(A.data(), m, n, X.data(), B.data(), tolerance, max_iterations);

    return 0;
}