if (AMReX_SPACEDIM GREATER_EQUAL 2)
    add_executable(test_cooling ../main.cpp test_cooling.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_cooling)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 2)
