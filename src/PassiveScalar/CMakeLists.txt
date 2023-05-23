add_executable(test_scalars test_scalars.cpp ../fextract.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_scalars)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME PassiveScalar COMMAND test_scalars PassiveScalar.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
