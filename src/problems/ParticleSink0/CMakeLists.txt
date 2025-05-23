if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_particle_sink0 test_particle_sink.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_particle_sink0)
    endif()

    add_test(NAME ParticleSink0 COMMAND test_particle_sink0 ParticleSink0.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
