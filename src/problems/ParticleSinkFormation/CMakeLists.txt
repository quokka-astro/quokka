if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_particle_sink_formation test_particle_sink_formation.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_particle_sink_formation)
    endif()

    add_test(NAME ParticleSinkFormation COMMAND test_particle_sink_formation ../inputs/ParticleSinkFormation.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
