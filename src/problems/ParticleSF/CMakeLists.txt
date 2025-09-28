if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_particle_SF test_particle_SF.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_particle_SF)
    endif()

    add_test(NAME ParticleSF COMMAND test_particle_SF ../inputs/ParticleSF.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
