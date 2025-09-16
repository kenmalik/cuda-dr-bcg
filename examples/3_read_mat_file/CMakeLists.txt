include(FetchContent)
FetchContent_Declare(
    suitesparse_matrix
    GIT_REPOSITORY git@github.com:kenmalik/suitesparse-matrix.git
    GIT_TAG main
)
FetchContent_MakeAvailable(suitesparse_matrix)

add_executable(
    read_mat_file
    "read_mat_file.cu"
)
target_link_libraries(read_mat_file PRIVATE dr_bcg suitesparse_matrix)

# Set RPATH to prioritize CUDA over MATLAB in conflicts
set_target_properties(
    read_mat_file PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${Matlab_ROOT_DIR}/bin/glnxa64"
    BUILD_WITH_INSTALL_RPATH TRUE
)
