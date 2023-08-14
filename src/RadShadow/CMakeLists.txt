
if (AMReX_SPACEDIM GREATER_EQUAL 2)
    add_executable(test_radiation_shadow test_radiation_shadow.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_radiation_shadow)
    endif()
endif()
