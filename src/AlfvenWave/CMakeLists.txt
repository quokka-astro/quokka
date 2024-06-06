add_executable(test_alfven_wave test_alfven_wave.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_alfven_wave)
endif()

add_test(NAME AlfvenWave COMMAND test_alfven_wave alfven_wave.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
