if (AMReX_SPACEDIM EQUAL 3)
set(JOB_NAME DiskGalaxy)

    add_executable(${JOB_NAME} testDiskGalaxy.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(${JOB_NAME})
    endif()
    
    # extract data files for regression testing
    file(ARCHIVE_EXTRACT INPUT ${QuokkaCode_SOURCE_DIR}/extern/agora_data/AgoraGalaxy_particles_LOW.tar.gz DESTINATION ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
    file(ARCHIVE_EXTRACT INPUT ${QuokkaCode_SOURCE_DIR}/extern/agora_data/vcirc.tar.gz DESTINATION ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
endif()
