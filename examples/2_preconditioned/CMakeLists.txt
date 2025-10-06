FetchContent_MakeAvailable(mat_utils)

add_executable(
    preconditioned
    "preconditioned.cu"
)
target_link_libraries(preconditioned PRIVATE dr_bcg mat_utils)

# Set RPATH to prioritize CUDA over MATLAB in conflicts
set_target_properties(
    preconditioned PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${Matlab_ROOT_DIR}/bin/glnxa64"
    BUILD_WITH_INSTALL_RPATH TRUE
)
