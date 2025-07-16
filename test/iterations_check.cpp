#include <tuple>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

std::vector<float> read_matrix_bin(const char *filename, const int num_elements)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        std::runtime_error("Failed to open file: " + std::string(filename));
    }

    std::vector<double> matrix_d(num_elements);
    std::vector<float> matrix_f(matrix_d.size());
    size_t read = fread(matrix_d.data(), sizeof(double), num_elements, f);
    fclose(f);

    if (read != num_elements)
    {
        std::runtime_error("Unexpected file size. Read " + std::to_string(read) + " elements");
    }

    std::transform(
        matrix_d.begin(), matrix_d.end(), matrix_f.begin(),
        [](double val)
        { return static_cast<float>(val); });

    return matrix_f;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: iterations_check [.bin file] [m]" << std::endl;
        return 1;
    }

    const char *filename = argv[1];
    const int m = std::stoi(argv[2]);

    constexpr int n = 4;
    constexpr float tolerance = 0.1;
    const int max_iterations = 2048;

    std::vector<float> A = read_matrix_bin(filename, m * m);
    std::vector<float> X_initial(m * n, 0);
    std::vector<float> B(m * n);
    fill_random(B.data(), m, n);

    auto [X_final, iterations] = dr_bcg::dr_bcg(A, X_initial, B, m, n, tolerance, max_iterations);

    std::cout << iterations << std::endl;

    return 0;
}
