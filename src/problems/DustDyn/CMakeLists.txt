add_executable(test_dust_dyn test_dust_dyn.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_dust_dyn)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME DustDyn COMMAND test_dust_dyn ../inputs/DustDyn.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
