if (AMReX_SPACEDIM EQUAL 3)
    add_executable(testMHDBitwiseICs testMHDBitwiseICs.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(testMHDBitwiseICs)
    endif()

    add_test(NAME MHDBitwiseICs COMMAND testMHDBitwiseICs ../inputs/MHDBitwiseICs.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(MHDBitwiseICs PROPERTIES LABELS "MHD")
endif()
