if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME ParticleSinkFormation)

  # Verify that the guard fires when sink particles + max_level >= 2 + do_subcycle=1 are combined.
  add_test(
    NAME ParticleSinkFormationSubcycleAbort
    COMMAND ParticleSinkFormation ../inputs/ParticleSinkFormation_amr2_subcycle.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(ParticleSinkFormationSubcycleAbort PROPERTIES
    PASS_REGULAR_EXPRESSION "ForceFinestLevel=true and max_level >= 2")
endif()
