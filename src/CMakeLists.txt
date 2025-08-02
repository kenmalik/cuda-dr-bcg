include(FetchContent)
FetchContent_Declare(
    nvtx
    GIT_REPOSITORY git@github.com:NVIDIA/NVTX.git
    GIT_TAG v3.2.1
)
FetchContent_MakeAvailable(nvtx)

add_library(dr_bcg
    "dr_bcg.cu"
    "helper.cu"
    "device_buffer.cu"
)
target_include_directories(dr_bcg PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(
    dr_bcg PUBLIC
    CUDA::cublas
    CUDA::cusolver
    nvtx3-cpp
)

target_compile_options(dr_bcg PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)