add_executable(test_dust_damping test_dust_damping.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_dust_damping)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME DustDamping COMMAND test_dust_damping ../inputs/DustDamping.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
