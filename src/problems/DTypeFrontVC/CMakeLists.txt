if(AMReX_SPACEDIM EQUAL 3)

  set(JOB_NAME DTypeFrontVC)

  # the photochemistry network: src/networks/photoionization_H_caseB_handwritten/network.hpp
  include_directories(BEFORE ${CMAKE_SOURCE_DIR}/src/networks/photoionization_H_caseB_handwritten)

  add_executable(
    ${JOB_NAME} testDTypeFrontVC.cpp "${QuokkaSourcesNoEOS}")

  # this will add #define CHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME DTypeFrontVC COMMAND ${JOB_NAME} ../inputs/DTypeFrontVC.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()