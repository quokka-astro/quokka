if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDAlfvenWaveLinear)
  set_tests_properties(MHDAlfvenWaveLinear PROPERTIES LABELS "MHD;MHD-ASAN")
endif()
