#include <iostream>
#include <vector>
#include <tuple>
#include <string>

#include "dr_bcg/helper.h"
#include "dr_bcg/dr_bcg.h"

int main(int argc, char *argv[])
{
    int m = 16;
    int n = 4;

    try {
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
    } catch (std::exception) {
        std::cerr << "Usage: ./example [m] [n]" << std::endl;
        std::cerr << "m and n values not specified, using defaults (m=" << m << ", n=" << n << ")" << std::endl;
    }

    constexpr float tolerance = 0.1;
    constexpr int max_iterations = 2048;

    std::vector<float> A(m * m);
    fill_spd(A.data(), m);
    std::vector<float> X(m * n, 0);
    std::vector<float> B(m * n);
    fill_random(B.data(), m, n);

    auto [solution, iterations] = dr_bcg::dr_bcg(A, X, B, m, n, tolerance, max_iterations);

    std::cout << "Solution X Final:" << std::endl;
    print_matrix(solution.data(), m, n);

    std::cout << "Done in " << iterations << " iterations" << std::endl;

    return 0;
}
