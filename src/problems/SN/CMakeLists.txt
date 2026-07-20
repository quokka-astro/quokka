if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME SN)

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  add_test(
    NAME ${JOB_NAME}
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(${JOB_NAME} PROPERTIES COST 90)

  add_test(
    NAME ${JOB_NAME}-n1.0
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml problem.n_amb=1.0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(${JOB_NAME}-n1.0 PROPERTIES COST 89)

  add_test(
    NAME ${JOB_NAME}-n0.1
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml problem.n_amb=0.1
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(${JOB_NAME}-n0.1 PROPERTIES COST 88)

  add_test(
    NAME ${JOB_NAME}-n10-no-smoothing
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml particles.SN_smooth_gas_velocity=false
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(${JOB_NAME}-n10-no-smoothing PROPERTIES COST 87)

  add_test(
    NAME ${JOB_NAME}-n1.0-no-smoothing
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml problem.n_amb=1.0 particles.SN_smooth_gas_velocity=false
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(${JOB_NAME}-n1.0-no-smoothing PROPERTIES COST 86)

  add_test(
    NAME ${JOB_NAME}-n0.1-no-smoothing
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml problem.n_amb=0.1 particles.SN_smooth_gas_velocity=false
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(${JOB_NAME}-n0.1-no-smoothing PROPERTIES COST 85)
endif()
