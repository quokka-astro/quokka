if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_orszag_tang test_orszag_tang.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_orszag_tang)
    endif()

    add_test(NAME OrszagTang COMMAND test_orszag_tang ../inputs/orszag_tang.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
