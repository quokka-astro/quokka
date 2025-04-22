if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_sink test_sink.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_sink)
    endif()

    add_test(NAME Sink COMMAND test_sink Sink.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
