set(JOB_NAME PrimordialChem)

add_executable(${JOB_NAME} testPrimordialChem.cpp "${QuokkaSourcesNoEOS}")
quokka_add_eos(${JOB_NAME} primordial_chem primordial_chem)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
