if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_field_loop test_field_loop.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_field_loop)
    endif()

    add_test(NAME FieldLoop COMMAND test_field_loop ../inputs/field_loop.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
