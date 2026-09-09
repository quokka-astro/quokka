if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "helium")
  set(DTypeFrontHe_INTEGRATOR "Rosenbrock" CACHE STRING "Microphysics integrator for DTypeFrontHe")
  set_property(CACHE DTypeFrontHe_INTEGRATOR PROPERTY STRINGS VODE Rosenbrock)
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/"
                                            INTEGRATOR "${DTypeFrontHe_INTEGRATOR}")

  set(JOB_NAME DTypeFrontHe)

  # NSPEC/SPECIES_ENUM/SPEC_NAMES/NUM_CHEM_BANDS/CHEM_BANDS/AION/... are derived
  # from jaff's own _parameters and jaffgen.toml by write_jaff_network_vars.py,
  # and network_properties.H is generated from them -- both done inside
  # setup_target_for_microphysics_compilation() above.

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${${microphysics_network_name}_sources})

  # this will add #define PHOTOCHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)
  if(DTypeFrontHe_INTEGRATOR STREQUAL "Rosenbrock")
    target_compile_definitions(${JOB_NAME} PUBLIC DTYPEFRONT_USE_ROSENBROCK)
  else()
    target_compile_definitions(${JOB_NAME} PUBLIC DTYPEFRONT_USE_VODE)
  endif()

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/DTypeFrontHe.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
