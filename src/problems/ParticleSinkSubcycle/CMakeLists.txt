if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME ParticleSinkSubcycle)
  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  # Test passes when the particle reaches the finest AMR level (level-2) under
  # do_subcycle=1 + max_level=2. On the development branch (no fix), the subcycled
  # level-1 regrid cannot expand level-1 to cover the particle, so level-2 is never
  # properly created and the test exits with code 1.
  add_test(
    NAME ParticleSinkSubcycle
    COMMAND ${JOB_NAME} ../inputs/ParticleSinkSubcycle_bug.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(ParticleSinkSubcycle PROPERTIES
    PASS_REGULAR_EXPRESSION "Test passed: particle is on finest level")
endif()