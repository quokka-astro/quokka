if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_typed_multifab test_typed_multifab.cpp ../fextract.cpp ${QuokkaObjSources})
    target_compile_definitions(test_typed_multifab PUBLIC ENABLE_HDF5)
    target_compile_definitions(test_typed_multifab PUBLIC ENABLE_OPENPMD)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_typed_multifab)
    endif()
endif()