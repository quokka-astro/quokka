if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_alfven_wave_circular test_alfven_wave_circular.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_alfven_wave_circular)
    endif()

    add_test(NAME AlfvenWaveCircular COMMAND test_alfven_wave_circular ../inputs/alfven_wave_circular.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(AlfvenWaveCircular PROPERTIES LABELS "MHD")
endif()
