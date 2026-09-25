# The problem is a spherical H II region around a central point source, so it is meaningful only in 3D.
if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "thermophotochemistry_handwritten")

  set(JOB_NAME DTypeFront3D)

  # NSPEC/SPECIES_ENUM/SPEC_NAMES/AION/ZION/CHEM_BANDS/NUM_THERMAL_BANDS/... come
  # from the 'thermophotochemistry_handwritten' entry in
  # src/networks/NetworkRegistry.cmake, and network_properties.H is generated
  # from them inside setup_target_for_microphysics_compilation() below.
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/"
                                            INTEGRATOR "Rosenbrock")

  add_executable(
    ${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${${microphysics_network_name}_sources})

  # this will add #define PHOTOCHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY SKIP_PHOTOCHEMFLUX THERMAL_DUST_PHOTOCHEMISTRY)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")
endif()
