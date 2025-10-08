if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_hydro_balsara_vortex test_hydro_balsara_vortex.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_hydro_balsara_vortex)
    endif()

    add_test(NAME HydroBalsaraVortex COMMAND test_hydro_balsara_vortex ../inputs/balsara_vortex.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
