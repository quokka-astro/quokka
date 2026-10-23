if(AMReX_SPACEDIM EQUAL 3)
  set(DTypeFront_JAFF_INTEGRATOR "Rosenbrock" CACHE STRING "Microphysics integrator for DTypeFront_JAFF (Rosenbrock or VODE)")
  set_property(CACHE DTypeFront_JAFF_INTEGRATOR PROPERTY STRINGS Rosenbrock VODE)
  set(DTypeFront_JAFF_C_HAT_DIVISOR 1000 CACHE STRING "DTypeFront_JAFF reduced speed of light: c_hat = c / divisor")
  option(DTypeFront_JAFF_NO_RADIATION "DTypeFront_JAFF without radiation transport (and so without photochemistry)" OFF)

  set(microphysics_network_name "photoionization_H_caseB")
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/"
                                            INTEGRATOR "${DTypeFront_JAFF_INTEGRATOR}")

  set(JOB_NAME DTypeFront_JAFF)

  # NSPEC/SPECIES_ENUM/SPEC_NAMES/NUM_CHEM_BANDS/CHEM_BANDS/AION/... are derived
  # from jaff's own _parameters and jaffgen.toml by write_jaff_network_vars.py,
  # and network_properties.H is generated from them -- both done inside
  # setup_target_for_microphysics_compilation() above.

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${${microphysics_network_name}_sources})

  # this will add #define PHOTOCHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY C_HAT_DIVISOR=${DTypeFront_JAFF_C_HAT_DIVISOR})
  if(DTypeFront_JAFF_INTEGRATOR STREQUAL "VODE")
    target_compile_definitions(${JOB_NAME} PUBLIC DTYPEFRONT_JAFF_VODE)
  endif()
  if(DTypeFront_JAFF_NO_RADIATION)
    target_compile_definitions(${JOB_NAME} PUBLIC DTYPEFRONT_JAFF_NO_RADIATION)
  endif()

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/DTypeFront_JAFF.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}RadPres COMMAND ${JOB_NAME} ../inputs/DTypeFrontRadPres.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
