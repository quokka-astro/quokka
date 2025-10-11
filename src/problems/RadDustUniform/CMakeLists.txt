if (AMReX_SPACEDIM EQUAL 3)
    add_executable(testRadDustUniform testRadDustUniform.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(testRadDustUniform)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM EQUAL 3)