if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_current_sheet test_current_sheet.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_current_sheet)
    endif()

    add_test(NAME CurrentSheet COMMAND test_current_sheet ../inputs/current_sheet.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
