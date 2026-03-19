if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDCurrentSheet)
  set_tests_properties(MHDCurrentSheet PROPERTIES LABELS "MHD-TEST")
endif()
