if(AMReX_SPACEDIM GREATER_EQUAL 2)
  if(AMReX_SPACEDIM EQUAL 2)
    set(disk_cells "amr.n_cell=256 256")
  else()
    set(disk_cells "amr.n_cell=256 256 8")
  endif()
  if(AMReX_SPACEDIM EQUAL 2)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/residual-test)
    add_test(NAME KeplerianDiskResidual
      COMMAND KeplerianDisk ${CMAKE_SOURCE_DIR}/inputs/KeplerianDisk.toml
        ${QuokkaTestParams} "amr.n_cell=256 256" max_timesteps=1 constant_dt=0.001 stop_time=0.001
        disk.residual_prefix=residual disk.test_tolerance=0.03
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/residual-test)
  endif()
  quokka_add_problem(JOB_NAME KeplerianDisk ADD_TEST OFF)
  add_executable(KeplerianDiskAlphaFit test_ring_alpha.cpp)
  target_compile_features(KeplerianDiskAlphaFit PRIVATE cxx_std_20)
  add_test(NAME KeplerianDiskAlphaFit COMMAND KeplerianDiskAlphaFit)
  add_test(NAME KeplerianDisk
    COMMAND KeplerianDisk ${CMAKE_SOURCE_DIR}/inputs/KeplerianDisk.toml
      ${QuokkaTestParams} "${disk_cells}" max_timesteps=2 disk.orbits=0.001 disk.check_transport=1 disk.check_initial=1 disk.check_alpha=1
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/power-law)
  add_test(NAME KeplerianDiskPowerLaw
    COMMAND KeplerianDisk ${CMAKE_SOURCE_DIR}/inputs/KeplerianDiskPowerLaw.toml
      ${QuokkaTestParams} "${disk_cells}" max_timesteps=2 constant_dt=0.001
      plotfile_interval=-1 disk.test_tolerance=0.03
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/power-law)
  foreach(radius IN ITEMS 40 60)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ring-${radius})
    add_test(NAME KeplerianDiskRing${radius}
      COMMAND KeplerianDisk ${CMAKE_SOURCE_DIR}/inputs/KeplerianDisk.toml
        ${QuokkaTestParams} "${disk_cells}" disk.ring_radius_cells=${radius}
        max_timesteps=0 disk.check_initial=1 disk.check_alpha=1
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ring-${radius})
  endforeach()
endif()

if(AMReX_SPACEDIM EQUAL 2)
  quokka_add_problem(JOB_NAME KeplerianDiskJoseph ADD_TEST OFF)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/joseph-test)
  add_test(NAME KeplerianDiskJoseph
    COMMAND KeplerianDiskJoseph ${CMAKE_SOURCE_DIR}/inputs/KeplerianDiskJoseph.toml
      ${QuokkaTestParams} "amr.n_cell=64 64" max_timesteps=4 checkpoint_interval=-1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/joseph-test)
endif()
