if (AMReX_SPACEDIM GREATER_EQUAL 2)
    add_executable(test_radiation_beam ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radiation_beam.cpp $<TARGET_OBJECTS:primordialchem>)
    
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_radiation_beam)
    endif()
endif()
