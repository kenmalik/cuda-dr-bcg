#include <iostream>
#include <vector>

#include "dr_bcg/helper.h"
#include "dr_bcg/dr-bcg.h"

int main(int argc, char *argv[])
{
    constexpr int m = 128;
    constexpr int n = 4;
    constexpr float tolerance = 0.01;
    constexpr int max_iterations = 128;

    std::vector<float> A(m * m);
    fill_spd(A.data(), m);
    std::vector<float> X(m * n, 0);
    std::vector<float> B(m * n);
    fill_random(B.data(), m, n);

    int iterations = dr_bcg::dr_bcg(A.data(), X.data(), B.data(), m, n, tolerance, max_iterations);

    std::cout << "Solution X Final:" << std::endl;
    print_matrix(X.data(), m, n);

    std::cout << "Done in " << iterations << " iterations" << std::endl;

    return 0;
}
