if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDBalsaraVortex)
  set_tests_properties(MHDBalsaraVortex PROPERTIES LABELS "MHD-TEST")
endif()
