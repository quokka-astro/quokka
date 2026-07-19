if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME ParticleRadiation)

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  add_test(
    NAME ${JOB_NAME}
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(
    NAME ParticleRadiationLog
    COMMAND ${JOB_NAME} ../inputs/ParticleRadiationLog.toml ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(
    NAME ParticleRadiationFastlog
    COMMAND ${JOB_NAME} ../inputs/ParticleRadiationFastlog.toml
            ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
