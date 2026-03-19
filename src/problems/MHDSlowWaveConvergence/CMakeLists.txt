if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDSlowWaveConvergence)
  set_tests_properties(MHDSlowWaveConvergence PROPERTIES LABELS "MHD-TEST")
endif()
