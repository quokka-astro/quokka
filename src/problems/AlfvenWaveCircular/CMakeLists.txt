if (AMReX_SPACEDIM EQUAL 3)
    add_executable(AlfvenWaveCircular AlfvenWaveCircular.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(AlfvenWaveCircular)
    endif()

    add_test(NAME AlfvenWaveCircular COMMAND AlfvenWaveCircular ../inputs/alfven_wave_circular.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(AlfvenWaveCircular PROPERTIES LABELS "MHD")
endif()
