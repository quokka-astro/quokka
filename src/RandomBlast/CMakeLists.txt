if (AMReX_SPACEDIM GREATER_EQUAL 3)
    add_executable(random_blast ../main.cpp blast.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp ../MFInterpolater.cpp)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(random_blast)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 3)