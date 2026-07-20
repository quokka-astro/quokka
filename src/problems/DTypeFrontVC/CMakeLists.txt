if(AMReX_SPACEDIM GREATER_EQUAL 2)
  set(JOB_NAME DTypeFrontVC)

  add_executable(
    ${JOB_NAME} testDTypeFrontVC.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp)

  # this will add #define CHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)
  target_link_libraries(${JOB_NAME} PRIVATE quokka::photoionization)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME DTypeFrontVC COMMAND ${JOB_NAME} ../inputs/DTypeFrontVC.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
