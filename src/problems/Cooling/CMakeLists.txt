if (AMReX_SPACEDIM GREATER_EQUAL 2)
set(JOB_NAME Cooling)

    add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(${JOB_NAME})
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 2)
