if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization_H_caseB")

  set(JOB_NAME DTypeFront_JAFF)

  # The Rosenbrock tableau of the problem's integrator (Microphysics_Traits<DTypeFront_JAFF>::Integrator), so that
  # scripts/slurm/dtypefront*_gpu_sweep.submit can time each tableau.
  set(DTypeFront_JAFF_TABLEAU "ROS2S" CACHE STRING "Rosenbrock tableau of DTypeFront_JAFF: ROS2S, Rodas3P, Rodas4P or Rodas5P")
  set_property(CACHE DTypeFront_JAFF_TABLEAU PROPERTY STRINGS ROS2S Rodas3P Rodas4P Rodas5P)
  if(NOT DTypeFront_JAFF_TABLEAU MATCHES "^(ROS2S|Rodas3P|Rodas4P|Rodas5P)$")
    message(FATAL_ERROR "DTypeFront_JAFF_TABLEAU must be ROS2S, Rodas3P, Rodas4P or Rodas5P, got '${DTypeFront_JAFF_TABLEAU}'")
  endif()

  set(DTypeFront_JAFF_C_HAT_DIVISOR 1000 CACHE STRING "DTypeFront_JAFF reduced speed of light: c_hat = c / divisor")
  option(DTypeFront_JAFF_NO_RADIATION "DTypeFront_JAFF without radiation transport (and so without photochemistry)" OFF)
  set(DTypeFront_JAFF_EXTRA_DEFINES "" CACHE STRING "Extra compile definitions for DTypeFront_JAFF (speed variants), e.g. ROSENBROCK_UNROLL_STAGES")

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
                ${CMAKE_SOURCE_DIR}/src/networks/${microphysics_network_name}/network.cpp)
  target_compile_definitions(${JOB_NAME} PRIVATE DTYPEFRONT_JAFF_TABLEAU=${DTypeFront_JAFF_TABLEAU} C_HAT_DIVISOR=${DTypeFront_JAFF_C_HAT_DIVISOR})
  if(DTypeFront_JAFF_NO_RADIATION)
    target_compile_definitions(${JOB_NAME} PRIVATE DTYPEFRONT_JAFF_NO_RADIATION)
  endif()
  if(DTypeFront_JAFF_EXTRA_DEFINES)
    target_compile_definitions(${JOB_NAME} PRIVATE ${DTypeFront_JAFF_EXTRA_DEFINES})
  endif()
  quokka_add_eos(${JOB_NAME} photoionization ${microphysics_network_name})
  # the network's actual_rhs.H
  target_include_directories(${JOB_NAME} BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/src/networks/${microphysics_network_name})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/DTypeFront_JAFF.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}RadPres COMMAND ${JOB_NAME} ../inputs/DTypeFrontRadPres.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
