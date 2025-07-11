if(AMReX_SPACEDIM EQUAL 3)
    add_executable(test_few_modes_ft test_few_modes_ft.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_few_modes_ft)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
endif()