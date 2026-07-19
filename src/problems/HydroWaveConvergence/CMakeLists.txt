set(JOB_NAME HydroWaveConvergence)

add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif()

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_test(
  NAME HydroWave
  COMMAND ${JOB_NAME} ../inputs/HydroWave.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_test(
  NAME HydroWaveFc
  COMMAND ${JOB_NAME} ../inputs/HydroWaveFc.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
