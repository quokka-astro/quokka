if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDOrszagTang)
  set_tests_properties(MHDOrszagTang PROPERTIES LABELS "MHD")
endif()
