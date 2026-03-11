if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDAlfvenWaveLinearConvergence)
  set_tests_properties(MHDAlfvenWaveLinearConvergence PROPERTIES LABELS "MHD")
endif()
