if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "dtype_photoionization_jaff") # this will override
                                                  # network_name to
                                                  # dtype_photoionization_jaff for this
                                                  # directory only
  set(DTypeFrontJaff_INTEGRATOR "Rosenbrock" CACHE STRING "Microphysics integrator for DTypeFrontJaff")
  set_property(CACHE DTypeFrontJaff_INTEGRATOR PROPERTY STRINGS VODE Rosenbrock)
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/"
                                            "${DTypeFrontJaff_INTEGRATOR}")

  # use the BEFORE keyword so that these files get priority in compilation for
  # targets in this directory this is critical to ensure the correct Microphysics
  # files are linked to photoionization target
  include_directories(
    BEFORE ${photoionization_dirs} "${CMAKE_CURRENT_BINARY_DIR}/"
    "includes/extern_parameters.H" "includes/network_properties.H")

  set(JOB_NAME DTypeFrontJaff)

  # NSPEC/SPECIES_ENUM/SPEC_NAMES/NUM_CHEM_BANDS/CHEM_BANDS are derived from
  # jaff's own _parameters and jaffgen.toml by write_jaff_network_vars.py
  # (invoked from setup_target_for_microphysics_compilation above), instead of
  # being hand-duplicated here.
  include(${jaff_network_vars_file})
  set(NEXTRASPEC 0)

  # No extra network properties
  set(PROPERTIES "")

  # Configure the header
  configure_file(
      ${CMAKE_SOURCE_DIR}/src/networks/network_header.template
      ${CMAKE_CURRENT_BINARY_DIR}/network_properties.H
      @ONLY
  )


  add_executable(
    ${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${photoionization_sources})

  # this will add #define CHEMISTRY
  # jaff's generated actual_rhs.H always computes the photon-flux equation
  # (ydot(6), state.rn[1]) unconditionally -- it has no SKIP_PHOTOCHEMFLUX guard
  # -- so this network cannot be built with SKIP_PHOTOCHEMFLUX defined.
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)
  if(DTypeFrontJaff_INTEGRATOR STREQUAL "Rosenbrock")
    target_compile_definitions(${JOB_NAME} PUBLIC DTYPEFRONT_USE_ROSENBROCK)
  else()
    target_compile_definitions(${JOB_NAME} PUBLIC DTYPEFRONT_USE_VODE)
  endif()

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  # Reuse DTypeFront's existing input files -- this problem is a jaff-generated
  # network swap-in, not a new physical setup.
  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/DTypeFront.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}RadPres COMMAND ${JOB_NAME} ../inputs/DTypeFrontRadPres.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
