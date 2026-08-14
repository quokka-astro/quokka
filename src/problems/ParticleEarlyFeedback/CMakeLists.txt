if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME ParticleEarlyFeedback)

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  add_test(
    NAME ${JOB_NAME}
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(
    NAME ${JOB_NAME}Boost
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml problem.boost_velocity=3.0e6 ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(
    NAME ${JOB_NAME}Cancellation
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml problem.inflow_speed=1.0e5 ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
