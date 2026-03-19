if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDFastWaveConvergence)
  set_tests_properties(MHDFastWaveConvergence PROPERTIES LABELS "MHD-TEST")
endif()
