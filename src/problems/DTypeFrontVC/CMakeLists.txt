if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization_H_caseB_handwritten")

  set(JOB_NAME DTypeFrontVC)

  add_executable(
    ${JOB_NAME} testDTypeFrontVC.cpp "${QuokkaSourcesNoEOS}"
                ${CMAKE_SOURCE_DIR}/src/networks/${microphysics_network_name}/network.cpp)
  quokka_add_eos(${JOB_NAME} photoionization ${microphysics_network_name})
  # the network's actual_rhs.H
  target_include_directories(${JOB_NAME} BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/src/networks/${microphysics_network_name})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME DTypeFrontVC COMMAND ${JOB_NAME} ../inputs/DTypeFrontVC.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()