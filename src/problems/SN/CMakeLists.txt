if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME SN)

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  add_test(
    NAME ${JOB_NAME}
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(
    NAME ${JOB_NAME}-n1.0
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in problem.n_amb=1.0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(
    NAME ${JOB_NAME}-n0.1
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in problem.n_amb=0.1
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
