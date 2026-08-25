set(JOB_NAME RadMarshakAsymptotic)

add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND STREQUAL "CUDA")

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_test(
  NAME MarshakWaveAsymptoticCorr
  COMMAND ${JOB_NAME} ../inputs/MarshakAsymptoticCorr.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
