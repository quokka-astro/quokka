if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME ParticleSinkSubcycle)
  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  # Test passes only when the PR #1961 guard fires:
  #   "Particles with ForceFinestLevel=true and max_level >= 2 require AMR subcycling
  #    to be disabled. Set do_subcycle = 0."
  # On development (no guard), the simulation runs and finestLevel() < max_level is observed.
  add_test(
    NAME ParticleSinkSubcycle
    COMMAND ${JOB_NAME} ../inputs/ParticleSinkSubcycle_bug.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(ParticleSinkSubcycle PROPERTIES
    PASS_REGULAR_EXPRESSION "ForceFinestLevel=true and max_level >= 2")
endif()
