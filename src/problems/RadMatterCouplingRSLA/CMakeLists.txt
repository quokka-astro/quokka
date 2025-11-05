set(JOB_NAME RadMatterCouplingRSLA)

add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
