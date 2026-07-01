if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME DiskGalaxy ADD_TEST OFF)

  # extract data files for regression testing
  file(
    ARCHIVE_EXTRACT INPUT
    ${QuokkaCode_SOURCE_DIR}/extern/agora_data/AgoraGalaxy_particles_LOW.tar.gz
    DESTINATION ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
  file(ARCHIVE_EXTRACT INPUT
       ${QuokkaCode_SOURCE_DIR}/extern/agora_data/vcirc.tar.gz DESTINATION
       ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
endif()
