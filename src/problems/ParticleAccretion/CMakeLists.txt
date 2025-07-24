if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_particle_accretion test_particle_accretion.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_particle_accretion)
    endif()

		add_test(NAME ParticleAccretion COMMAND test_particle_accretion ../inputs/ParticleAccretion.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
		add_test(NAME ParticleAccretionBH COMMAND test_particle_accretion ../inputs/ParticleAccretionBH.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
