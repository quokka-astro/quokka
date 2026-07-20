if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME FastWaveConvergence)

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  add_test(
    NAME ${JOB_NAME}
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(FastWaveConvergence PROPERTIES LABELS "MHD")

  add_test(
    NAME FastWave
    COMMAND ${JOB_NAME} ../inputs/FastWave.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(FastWave PROPERTIES LABELS "MHD")
endif()