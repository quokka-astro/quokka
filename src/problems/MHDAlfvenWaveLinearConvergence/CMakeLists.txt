if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDAlfvenWaveLinearConvergence)
  set_tests_properties(MHDAlfvenWaveLinearConvergence PROPERTIES LABELS "MHD")

  add_test(
    NAME MHDAlfvenWaveLinear
    COMMAND MHDAlfvenWaveLinearConvergence ../inputs/MHDAlfvenWaveLinear.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(MHDAlfvenWaveLinear PROPERTIES LABELS "MHD")
endif()
