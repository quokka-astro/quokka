quokka_add_problem(JOB_NAME RadStreaming)

# Same executable, run with a beam injected by the reducedFluxSource hook instead of the boundary
add_test(
  NAME RadStreamingFluxSource
  COMMAND RadStreaming ../inputs/RadStreamingFluxSource.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
