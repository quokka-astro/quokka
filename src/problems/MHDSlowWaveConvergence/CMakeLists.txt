if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDSlowWaveConvergence)
  set_tests_properties(MHDSlowWaveConvergence PROPERTIES LABELS "MHD")

  add_test(
    NAME MHDSlowWave
    COMMAND MHDSlowWaveConvergence ../inputs/MHDSlowWave.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(MHDSlowWave PROPERTIES LABELS "MHD")
endif()
