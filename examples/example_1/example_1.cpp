#include <tuple>
#include <iostream>
#include <vector>
#include <string>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

void read_matrix_bin(double *buffer, const char *filename, const int num_elements)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        std::runtime_error("Failed to open file: " + std::string(filename));
    }

    size_t read = fread(buffer, sizeof(double), num_elements, f);
    fclose(f);

    if (read != num_elements)
    {
        std::runtime_error("Unexpected file size. Read " + std::to_string(read) + " elements");
    }
}

std::vector<float> d_to_s(const std::vector<double> &vec_d)
{
    std::vector<float> vec_s(vec_d.size());

    for (size_t i = 0; i < vec_d.size(); i++)
    {
        vec_s.at(i) = static_cast<float>(vec_d.at(i));
    }

    return vec_s;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Example 2 reads an SPD matrix in a .bin file and runs "
                  << "dr_bcg on it with a random B" << std::endl;
        std::cerr << "Usage: ./example_2 [.bin file] [m]" << std::endl;
        return 1;
    }

    const char *filename = argv[1];
    const size_t m = std::stoi(argv[2]);
    constexpr int n = 4;

    constexpr float tolerance = 0.1;
    constexpr int max_iterations = 100;

    std::vector<double> matrix_d(m * m);
    read_matrix_bin(matrix_d.data(), filename, matrix_d.size());

    std::vector<float> A = d_to_s(matrix_d);
    std::vector<float> X_initial(m * n, 0);
    std::vector<float> B(m * n);
    fill_random(B.data(), m, n);

    auto [X_final, iterations] = dr_bcg::dr_bcg(A, X_initial, B, m, n, tolerance, max_iterations);

    std::cout << "Final X:" << std::endl;
    print_matrix(X_final.data(), m, n);
    std::cout << "Done in " << iterations << " iterations" << std::endl;

    return 0;
}
