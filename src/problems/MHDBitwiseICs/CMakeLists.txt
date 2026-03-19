if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDBitwiseICs)
  set_tests_properties(MHDBitwiseICs PROPERTIES LABELS "MHD-TEST;MHD-ASAN")
endif()
