include(FetchContent)
FetchContent_Declare(
    suitesparse_matrix
    GIT_REPOSITORY git@github.com:kenmalik/suitesparse-matrix.git
    GIT_TAG main
)
FetchContent_MakeAvailable(suitesparse_matrix)

add_executable(
    example_2
    "example_2.cu"
)
target_link_libraries(example_2 PRIVATE dr_bcg suitesparse_matrix)

# Set RPATH to prioritize CUDA over MATLAB in conflicts
set_target_properties(
    example_2 PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${Matlab_ROOT_DIR}/bin/glnxa64"
    BUILD_WITH_INSTALL_RPATH TRUE
)
