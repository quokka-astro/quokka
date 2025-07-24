if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_alfven_wave_linear test_alfven_wave_linear.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_alfven_wave_linear)
    endif()

    add_test(NAME AlfvenWaveLinear COMMAND test_alfven_wave_linear ../inputs/alfven_wave_linear.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(AlfvenWaveLinear PROPERTIES LABELS "MHD")
endif()
