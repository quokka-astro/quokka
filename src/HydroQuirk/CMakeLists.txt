if (AMReX_SPACEDIM GREATER_EQUAL 2)
    add_executable(test_quirk ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_quirk.cpp)
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_quirk)
    endif()

    add_test(NAME HydroQuirk COMMAND test_quirk quirk.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
