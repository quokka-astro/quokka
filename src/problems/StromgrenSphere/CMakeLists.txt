if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization_H_caseB_handwritten")

  set(JOB_NAME StromgrenSphere)

  # two executables built from StromgrenSphere.hpp, each with its own test and input
  foreach(target ${JOB_NAME} ${JOB_NAME}TempIndependentRecombination)
    add_executable(
      ${target} test${target}.cpp "${QuokkaSourcesNoEOS}"
                ${CMAKE_SOURCE_DIR}/src/networks/${microphysics_network_name}/network.cpp)
    quokka_add_eos(${target} photoionization ${microphysics_network_name})
    # the network's actual_rhs.H
    target_include_directories(${target} BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/src/networks/${microphysics_network_name})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(${target})
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")

    add_test(NAME ${target} COMMAND ${target} ../inputs/${target}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  endforeach()
endif()