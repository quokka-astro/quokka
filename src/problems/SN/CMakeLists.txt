if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_SN test_SN.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_SN)
    endif()

    add_test(NAME SN COMMAND test_SN SN.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
