if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDAlfvenWaveCircularConvergence)
  set_tests_properties(MHDAlfvenWaveCircularConvergence PROPERTIES LABELS "MHD")

  add_test(
    NAME MHDAlfvenWaveCircular
    COMMAND MHDAlfvenWaveCircularConvergence ../inputs/MHDAlfvenWaveCircular.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(MHDAlfvenWaveCircular PROPERTIES LABELS "MHD")
endif()
