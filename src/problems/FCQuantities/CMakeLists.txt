if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME FCQuantities)
  set_tests_properties(FCQuantities PROPERTIES LABELS "MHD-TEST;MHD-ASAN")
endif()
