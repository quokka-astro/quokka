if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_fast_wave test_fast_wave.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_fast_wave)
    endif()

    add_test(NAME FastWave COMMAND test_fast_wave ../inputs/fast_wave.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(FastWave PROPERTIES LABELS "MHD")
endif()
