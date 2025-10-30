add_executable(test_dusty_shock test_dusty_shock.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_dusty_shock)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME DustyShock COMMAND test_dusty_shock ../inputs/DustyShock.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
