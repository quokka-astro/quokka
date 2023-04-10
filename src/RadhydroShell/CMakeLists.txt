if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_radhydro3d_shell ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radhydro_shell.cpp ../interpolate.cpp $<TARGET_OBJECTS:primordialchem>)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_radhydro3d_shell)
    endif()
endif()
