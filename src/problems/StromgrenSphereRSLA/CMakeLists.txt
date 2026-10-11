if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization_H_caseB_handwritten") # this will override
                                                  # network_name to
                                                  # photoionization for this
                                                  # directory only

  set(JOB_NAME StromgrenSphereRSLA)

  # NSPEC/SPECIES_ENUM/SPEC_NAMES/AION/ZION/CHEM_BANDS/... come from the
  # 'photoionization' entry in src/networks/NetworkRegistry.cmake, and
  # network_properties.H is generated from them inside
  # setup_target_for_microphysics_compilation() below.
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/"
                                            INTEGRATOR "Rosenbrock")

  add_executable(
    ${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${${microphysics_network_name}_sources})

  # this will add #define CHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
