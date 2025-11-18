if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME HydroBlast3D INPUT_FILE blast_unigrid_128.in)
endif()
