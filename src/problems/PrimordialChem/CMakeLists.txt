set(JOB_NAME PrimordialChem)

add_executable(
  ${JOB_NAME} testPrimordialChem.cpp "${QuokkaSourcesNoEOS}"
              ../../chemistry/Chemistry.cpp)

# this will add #define CHEMISTRY
target_compile_definitions(${JOB_NAME} PUBLIC CHEMISTRY)
target_link_libraries(${JOB_NAME} PRIVATE quokka::primordial_chem)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
