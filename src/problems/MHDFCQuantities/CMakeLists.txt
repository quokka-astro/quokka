if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDFCQuantities)
  set_tests_properties(MHDFCQuantities PROPERTIES LABELS "MHD-TEST;MHD-ASAN")
endif()
