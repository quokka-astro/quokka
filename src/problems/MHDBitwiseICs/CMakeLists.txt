if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_mhd_bitwise_ICs test_mhd_bitwise_ICs.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_mhd_bitwise_ICs)
    endif()

    add_test(NAME MHDBitwiseICs COMMAND test_mhd_bitwise_ICs ../inputs/alfven_wave_linear_regression.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(MHDBitwiseICs PROPERTIES LABELS "MHD")
endif()
