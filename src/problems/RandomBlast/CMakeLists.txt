if (AMReX_SPACEDIM EQUAL 3)
set(JOB_NAME RandomBlast)

    add_executable(${JOB_NAME} testRandomBlast.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(${JOB_NAME})
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM EQUAL 3)