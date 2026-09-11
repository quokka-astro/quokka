if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDEntropyWaveConvergence)
  set_tests_properties(MHDEntropyWaveConvergence PROPERTIES LABELS "MHD")

  add_test(
    NAME MHDEntropyWave
    COMMAND MHDEntropyWaveConvergence ../inputs/MHDEntropyWave.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(MHDEntropyWave PROPERTIES LABELS "MHD")
endif()