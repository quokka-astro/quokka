if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME MHDBrioWuShockTube)
  set_tests_properties(MHDBrioWuShockTube PROPERTIES LABELS "MHD-TEST")
endif()
