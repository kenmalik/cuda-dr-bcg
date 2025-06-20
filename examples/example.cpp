#include <iostream>
#include <vector>

#include "dr_bcg/helper.h"
#include "dr_bcg/dr-bcg.h"

int main(int argc, char *argv[])
{
    constexpr int m = 8;
    constexpr int n = 4;
    constexpr float tolerance = 0.001;
    constexpr int max_iterations = 25;

    std::vector<float> A(m * m);
    fill_spd(A.data(), m);
    std::vector<float> X(m * n);
    fill_random(X.data(), m, n);
    std::vector<float> B(m * n);
    fill_random(B.data(), m, n);

    std::cout << "Before:" << std::endl;
    std::cout << "A:" << std::endl;
    print_matrix(A.data(), m, m);
    std::cout << "X:" << std::endl;
    print_matrix(X.data(), m, n);
    std::cout << "B:" << std::endl;
    print_matrix(B.data(), m, n);

    dr_bcg::dr_bcg(A.data(), m, n, X.data(), B.data(), tolerance, max_iterations);

    return 0;
}