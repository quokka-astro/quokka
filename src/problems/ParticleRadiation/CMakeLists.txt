if (AMReX_SPACEDIM EQUAL 3)
    add_executable(particle_radiation particle_radiation.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(particle_radiation)
    endif()

    add_test(NAME ParticleRadiation COMMAND particle_radiation ../inputs/ParticleRadiation.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
