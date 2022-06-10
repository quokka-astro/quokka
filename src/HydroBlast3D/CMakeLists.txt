if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_hydro3d_blast ../main.cpp test_hydro3d_blast.cpp)
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_hydro3d_blast)
    endif()

    add_test(NAME HydroBlast3D COMMAND test_hydro3d_blast blast_unigrid_128.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
