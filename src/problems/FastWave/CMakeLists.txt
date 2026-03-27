if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME FastWave)
  set_tests_properties(FastWave PROPERTIES LABELS "MHD-TEST")
endif()
