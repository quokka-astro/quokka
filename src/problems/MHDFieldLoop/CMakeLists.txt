if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDFieldLoop)
  set_tests_properties(MHDFieldLoop PROPERTIES LABELS "MHD")
endif()
