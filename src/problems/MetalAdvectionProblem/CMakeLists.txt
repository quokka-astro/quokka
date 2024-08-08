if (AMReX_SPACEDIM EQUAL 3)
      add_executable(test_sne test_sne.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_sne)
    endif()

endif()
