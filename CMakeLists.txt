cmake_minimum_required(VERSION 3.17)
project(DR_BCG VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(dr-bcg dr-bcg.cu)
target_compile_options(dr-bcg PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_include_directories(dr-bcg PRIVATE "cutlass/include" "cutlass/tools/util/include")

target_link_libraries(dr-bcg PRIVATE CUDA::cublas CUDA::cusolver)
