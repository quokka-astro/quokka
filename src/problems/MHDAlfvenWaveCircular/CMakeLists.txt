if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDAlfvenWaveCircular)
  set_tests_properties(MHDAlfvenWaveCircular PROPERTIES LABELS "MHD;MHD-ASAN")
endif()
