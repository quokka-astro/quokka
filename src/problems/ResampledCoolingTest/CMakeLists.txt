add_executable(test_resampled_cooling test_resampled_cooling.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_resampled_cooling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ResampledCoolingTest COMMAND test_resampled_cooling ../inputs/ResampledCooling.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
