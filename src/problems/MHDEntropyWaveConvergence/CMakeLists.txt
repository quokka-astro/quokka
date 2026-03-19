if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDEntropyWaveConvergence)
  set_tests_properties(MHDEntropyWaveConvergence PROPERTIES LABELS "MHD-TEST")
endif()
