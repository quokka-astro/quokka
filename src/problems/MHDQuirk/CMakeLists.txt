if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_mhd_quirk test_mhd_quirk.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_mhd_quirk)
    endif()

    add_test(NAME MHDQuirk COMMAND test_mhd_quirk ../inputs/quirk.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
		set_tests_properties(MHDQuirk PROPERTIES LABELS "MHD")
endif()
