if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDFastWaveConvergence)
  set_tests_properties(MHDFastWaveConvergence PROPERTIES LABELS "MHD")

  add_test(
    NAME MHDFastWave
    COMMAND MHDFastWaveConvergence ../inputs/MHDFastWave.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(MHDFastWave PROPERTIES LABELS "MHD")
endif()