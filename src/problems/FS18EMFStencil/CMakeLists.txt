if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME FS18EMFStencil)
  set_tests_properties(FS18EMFStencil PROPERTIES LABELS "MHD-TEST;MHD-ASAN")
endif()
