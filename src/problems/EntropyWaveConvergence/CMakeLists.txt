if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME EntropyWaveConvergence)
  set_tests_properties(EntropyWaveConvergence PROPERTIES LABELS "MHD")
endif()
