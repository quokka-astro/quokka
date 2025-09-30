if(AMReX_SPACEDIM EQUAL 3)
    add_executable(test_turb_driving test_turb_driving.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_turb_driving)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
endif()
