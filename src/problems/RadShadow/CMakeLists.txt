
if (AMReX_SPACEDIM GREATER_EQUAL 2)
set(JOB_NAME RadShadow)

    add_executable(${JOB_NAME} testRadShadow.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(${JOB_NAME})
    endif()
endif()
