add_executable(test_dust_soundwave test_dust_soundwave.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_dust_soundwave)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME DustSoundwave COMMAND test_dust_soundwave ../inputs/DustSoundwave.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)