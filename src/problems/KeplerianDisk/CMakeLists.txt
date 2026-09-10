if(AMReX_SPACEDIM GREATER_EQUAL 2)
  if(AMReX_SPACEDIM EQUAL 2)
    set(disk_cells "amr.n_cell=64 64")
  else()
    set(disk_cells "amr.n_cell=64 64 8")
  endif()
  quokka_add_problem(JOB_NAME KeplerianDisk ADD_TEST OFF)
  add_test(NAME KeplerianDisk
    COMMAND KeplerianDisk ${CMAKE_SOURCE_DIR}/inputs/KeplerianDisk.toml
      ${QuokkaTestParams} "${disk_cells}" stop_time=0.25 disk.test_tolerance=0.03 disk.check_transport=1
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
