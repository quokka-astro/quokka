if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_fast_wave_perm3 test_fast_wave_perm3.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_fast_wave_perm3)
    endif()

    add_test(NAME FastWave3 COMMAND test_fast_wave_perm3 fast_wave.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
