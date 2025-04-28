if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_particle_sink test_particle_sink.cpp ../../util/fextract.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_particle_sink)
    endif()

    add_test(NAME ParticleSink COMMAND test_particle_sink ParticleSink.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
