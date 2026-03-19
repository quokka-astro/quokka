if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDFastWave)
  set_tests_properties(MHDFastWave PROPERTIES LABELS "MHD-TEST")
endif()
