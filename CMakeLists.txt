cmake_minimum_required(VERSION 3.17)
project(DR_BCG VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_library(dr_bcg
    "src/dr-bcg.cu"
    "src/helper.cu"
)
target_include_directories(dr_bcg PUBLIC "include")
target_link_libraries(dr_bcg PRIVATE CUDA::cublas CUDA::cusolver)
target_compile_options(dr_bcg PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(example
    "examples/example.cpp"
)
target_link_libraries(example PRIVATE dr_bcg)