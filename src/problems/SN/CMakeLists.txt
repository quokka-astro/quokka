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
    NAME SN_long_grackle
    COMMAND ${JOB_NAME} ../inputs/SN_long_grackle.in
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(
    NAME SN_long_resampled
    COMMAND ${JOB_NAME} ../inputs/SN_long_resampled.in
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(
    NAME SN_feedback_only
    COMMAND ${JOB_NAME} ../inputs/SN_feedback_only.in
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
