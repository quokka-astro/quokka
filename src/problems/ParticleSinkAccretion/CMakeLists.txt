if (AMReX_SPACEDIM EQUAL 3)
    add_executable(particle_sink_accretion particle_sink_accretion.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(particle_sink_accretion)
    endif()

		add_test(NAME ParticleSinkAccretion COMMAND particle_sink_accretion ParticleSinkAccretion.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
