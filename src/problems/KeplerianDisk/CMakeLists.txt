if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME KeplerianDisk ADD_TEST OFF)

  # Extract the DiskGalaxy rotation curve used by the example input.
  file(ARCHIVE_EXTRACT INPUT
       ${QuokkaCode_SOURCE_DIR}/extern/agora_data/vcirc.tar.gz DESTINATION
       ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
endif()
