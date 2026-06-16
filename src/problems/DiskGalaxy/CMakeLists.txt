if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME DiskGalaxy ADD_TEST OFF)

  add_executable(DiskGalaxyOverwriteTemperature overwrite_temperature.cpp ../../cooling/ResampledCooling.cpp)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(DiskGalaxyOverwriteTemperature)
  endif()

  # extract data files for regression testing
  file(
    ARCHIVE_EXTRACT INPUT
    ${QuokkaCode_SOURCE_DIR}/extern/agora_data/AgoraGalaxy_particles_LOW.tar.gz
    DESTINATION ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
  file(ARCHIVE_EXTRACT INPUT
       ${QuokkaCode_SOURCE_DIR}/extern/agora_data/vcirc.tar.gz DESTINATION
       ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
endif()
