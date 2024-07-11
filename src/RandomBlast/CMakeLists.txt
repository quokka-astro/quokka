if (AMReX_SPACEDIM EQUAL 3)
    add_executable(random_blast blast.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(random_blast)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM EQUAL 3)