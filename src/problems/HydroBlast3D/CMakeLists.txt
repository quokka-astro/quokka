if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME HydroBlast3D)

  add_test(
    NAME HydroBlast3DViscous
    COMMAND HydroBlast3D ../inputs/HydroBlast3DViscous.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
