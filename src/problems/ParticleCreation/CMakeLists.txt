if (AMReX_SPACEDIM EQUAL 3)
    add_executable(particle_creation particle_creation_from_cell.cpp ../../math/interpolate.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(particle_creation)
    endif()

    add_test(NAME ParticleCreation COMMAND particle_creation ParticleCreation.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME ParticleCreationAMR COMMAND particle_creation ParticleCreationAMR.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
