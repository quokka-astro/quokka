if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_random_blast_rad test_random_blast_rad.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_random_blast_rad)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM EQUAL 3)