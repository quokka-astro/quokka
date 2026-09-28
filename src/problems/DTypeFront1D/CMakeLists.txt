# The problem is planar, so it runs unchanged in 3D: the slab is uniform across y and z and every transverse
# column repeats the 1D solution, which is what makes it a useful check that the radiation, dust and
# photochemistry paths behave identically in 3D. 2D is skipped only to keep CI time down.
if(AMReX_SPACEDIM EQUAL 1 OR AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization_H_caseB_handwritten") # this will override
                                                   # network_name to
                                                   # photoionization for this
                                                   # directory only

  set(JOB_NAME DTypeFront1D)

  # NSPEC/SPECIES_ENUM/SPEC_NAMES/AION/ZION/CHEM_BANDS/... come from the
  # 'photoionization_H_caseB_handwritten' entry in
  # src/networks/NetworkRegistry.cmake, and network_properties.H is generated
  # from them inside setup_target_for_microphysics_compilation() below.
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/"
                                            INTEGRATOR "Rosenbrock")

  add_executable(
    ${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${${microphysics_network_name}_sources})

  # this will add #define PHOTOCHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY SKIP_PHOTOCHEMFLUX)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  # One input serves both dimensionalities; only the x entries matter in 1D.
  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
