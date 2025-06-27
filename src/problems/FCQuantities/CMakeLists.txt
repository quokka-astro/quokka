if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_fc_quantities test_fc_quantities.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_fc_quantities)
    endif()

    add_test(NAME FCQuantities COMMAND test_fc_quantities ../inputs/fc_hydro_wave.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(FCQuantities PROPERTIES LABELS "MHD")
endif()
