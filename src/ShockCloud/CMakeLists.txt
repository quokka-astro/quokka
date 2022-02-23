if (AMReX_SPACEDIM GREATER_EQUAL 3)
    add_executable(shock_cloud ../main.cpp cloud.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(shock_cloud)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 3)