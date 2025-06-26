cmake_minimum_required(VERSION 3.17)
project(DR_BCG VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

option(DR_BCG_BUILD_TESTS "Enable tests" OFF)
option(DR_BCG_BUILD_BENCHMARKS "Enable benchmarks" OFF)
option(DR_BCG_BUILD_EXAMPLES "Enable examples" OFF)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_library(dr_bcg
    "src/dr_bcg.cu"
    "src/helper.cu"
)
target_include_directories(dr_bcg PUBLIC "include")
target_link_libraries(dr_bcg PUBLIC CUDA::cublas CUDA::cusolver)
target_compile_options(dr_bcg PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(DR_BCG_BUILD_TESTS)
    add_subdirectory("test")
endif()

if(DR_BCG_BUILD_BENCHMARKS)
    add_subdirectory("benchmark")
endif()

if(DR_BCG_BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()
