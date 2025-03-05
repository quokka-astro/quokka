if (AMReX_SPACEDIM EQUAL 3)
    add_executable(spherical_collapse spherical_collapse.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(spherical_collapse)
    endif()

    if(AMReX_GPU_BACKEND MATCHES "HIP")
        # This test failed when compiled with HIP/ROCm on the moth computer with a FPE error: Erroneous arithmetic operation. This is likely due to a bug in the compiler, so we are disabling FPE checks for this test on moth.
        add_test(NAME SphericalCollapse COMMAND spherical_collapse SphericalCollapse.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    else()
        add_test(NAME SphericalCollapse COMMAND spherical_collapse SphericalCollapse.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    endif()
endif()
