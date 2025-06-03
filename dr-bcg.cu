#include <iostream>

#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/layout/layout.h>

// A: n*n
// x: n*1
// b: n*1
int main(int argc, char *argv[]) {
    constexpr int n = 64;

    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> A({n, n});
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> b({n, 1});
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> x({n, 1});

    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 1, 4, -4, 0);
    cutlass::reference::host::TensorFillRandomUniform(x.host_view(), 1, 4, -4, 0);
    cutlass::reference::host::TensorFillRandomUniform(b.host_view(), 1, 4, -4, 0);

    std::cout << "A:\n" << A.host_view() << std::endl;
    std::cout << "x:\n" << x.host_view() << std::endl;
    std::cout << "b:\n" << b.host_view() << std::endl;
}