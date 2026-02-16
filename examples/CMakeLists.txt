add_executable(
    simple_sparse_double
    "simple_sparse_double.cu"
)

target_link_libraries(
    simple_sparse_double
    DR_BCG::dr_bcg
)

target_compile_features(simple_sparse_double PRIVATE cxx_std_17)
