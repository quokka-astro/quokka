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
                ${${microphysics_network_name}_sources}
                ${CMAKE_SOURCE_DIR}/src/networks/${microphysics_network_name}/network.cpp)

  # this will add #define PHOTOCHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY SKIP_PHOTOCHEMFLUX DUST_CHEMICAL_BAND_ABSORPTION)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  # Too expensive for the CPU-only GitHub Actions CI, so it is registered only in GPU builds.
  # plotfile_interval=-1 on the command line, because evolve() re-reads the inputs file and would undo a change made in code.
  if(AMReX_GPU_BACKEND MATCHES "CUDA|HIP")
    add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams} plotfile_interval=-1
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(${JOB_NAME} PROPERTIES TIMEOUT 5400)
  endif()
endif()
