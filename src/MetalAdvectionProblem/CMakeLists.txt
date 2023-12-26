if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_sne ../main.cpp test_sne.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp)
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_sne)
    endif()

    add_test(NAME HydroBlast3D COMMAND test_sne blast_unigrid_128.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()