if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME ParticleSinkSubcycle)
  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  # Verify that sink particles reach the finest AMR level (level-2) under
  # do_subcycle=1 + max_level=2.  This is the regression test for issue #1957.
  add_test(
    NAME ParticleSinkSubcycle
    COMMAND ${JOB_NAME} ../inputs/ParticleSinkSubcycle_bug.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(ParticleSinkSubcycle PROPERTIES
    PASS_REGULAR_EXPRESSION "Test passed: particle is on finest level")

  # Also verify do_subcycle=0 with max_level=2.
  add_test(
    NAME ParticleSinkSubcycleNoSubcycle
    COMMAND ${JOB_NAME} ../inputs/ParticleSinkSubcycle_bug.toml do_subcycle=0 ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(ParticleSinkSubcycleNoSubcycle PROPERTIES
    PASS_REGULAR_EXPRESSION "Test passed: particle is on finest level")

  # Pure particle-driven refinement: no density tags, ForceFinestLevel alone
  # drives all AMR levels.  Verifies that the force-regrid + cross-level tagging
  # create the full hierarchy without density-based refinement help.
  add_test(
    NAME ParticleSinkSubcycleParticleOnly
    COMMAND ${JOB_NAME} ../inputs/ParticleSinkSubcycle_particle_only.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(ParticleSinkSubcycleParticleOnly PROPERTIES
    PASS_REGULAR_EXPRESSION "Test passed: particle is on finest level")
endif()