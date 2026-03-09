if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME ViscousSedovShock)
  set_tests_properties(ViscousSedovShock PROPERTIES LABELS "Hydro")
endif()
