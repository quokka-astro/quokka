if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_particle_isothermal test_particle_isothermal.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_particle_isothermal)
    endif()

		add_test(NAME ParticleIsothermal COMMAND test_particle_isothermal ../inputs/ParticleIsothermal.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
