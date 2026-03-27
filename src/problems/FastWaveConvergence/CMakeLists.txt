if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME FastWaveConvergence)
  set_tests_properties(FastWaveConvergence PROPERTIES LABELS "MHD-TEST")
endif()
