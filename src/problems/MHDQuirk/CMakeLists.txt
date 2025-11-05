if (AMReX_SPACEDIM EQUAL 3)
set(JOB_NAME MHDQuirk)

    add_executable(${JOB_NAME} testMHDQuirk.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(${JOB_NAME})
    endif()

    add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
		set_tests_properties(MHDQuirk PROPERTIES LABELS "MHD")
endif()
