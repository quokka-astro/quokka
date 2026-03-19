if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDQuirk)
  set_tests_properties(MHDQuirk PROPERTIES LABELS "MHD-TEST")
endif()
