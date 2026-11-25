set(microphysics_network_name "thermophotochemistry_ism_handwritten")

set(JOB_NAME ISMCoolingOneZone)

# NSPEC/SPECIES_ENUM/SPEC_NAMES/AION/ZION/CHEM_BANDS/... come from the
# 'thermophotochemistry_ism_handwritten' entry in src/networks/NetworkRegistry.cmake, and
# network_properties.H is generated from them inside
# setup_target_for_microphysics_compilation() below.
setup_target_for_microphysics_compilation(${microphysics_network_name}
                                          "${CMAKE_CURRENT_BINARY_DIR}/"
                                          INTEGRATOR "Rosenbrock")

add_executable(
  ${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
              ../../radiation/photochemistry.cpp ${${microphysics_network_name}_sources})

target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY PHOTOCHEM_ISM_COOLING DUST_CHEMICAL_BAND_ABSORPTION)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ISMCoolingJacobian COMMAND ${JOB_NAME} ../inputs/ISMCoolingJacobian.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME ISMCoolingStartup COMMAND ${JOB_NAME} ../inputs/ISMCoolingStartup.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# 1 Gyr of 25 cells at fixed n_H against the Python thermal equilibrium (extern/cooling/ism_equilibrium_reference.csv); 1D only
if(AMReX_SPACEDIM EQUAL 1)
  add_test(NAME ISMCoolingEquilibrium COMMAND ${JOB_NAME} ../inputs/ISMCoolingEquilibrium.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  # one step of the hot-cell cooling against the Python integration (extern/cooling/ism_hot_cell_reference.csv), with dust-only attenuation
  add_test(NAME ISMCoolingHotCell COMMAND ${JOB_NAME} ../inputs/ISMCoolingHotCell.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  # a cell cools through T_th = 2e4 K in the hot-cell step and recombines in the burn on the next step
  add_test(NAME ISMCoolingCrossThreshold COMMAND ${JOB_NAME} ../inputs/ISMCoolingCrossThreshold.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()

# the startup of the photochemistry cooling path aborts, naming the parameter, without G0_external, without the table path and with a table of
# another metallicity; PASS_REGULAR_EXPRESSION makes each pass only for its own abort message
add_test(NAME ISMCoolingNoG0 COMMAND ${JOB_NAME} ../inputs/ISMCoolingNoG0.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME ISMCoolingNoTable COMMAND ${JOB_NAME} ../inputs/ISMCoolingNoTable.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME ISMCoolingZMismatch COMMAND ${JOB_NAME} ../inputs/ISMCoolingZMismatch.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
set_tests_properties(ISMCoolingNoG0 PROPERTIES PASS_REGULAR_EXPRESSION "G0_external is not set")
set_tests_properties(ISMCoolingNoTable PROPERTIES PASS_REGULAR_EXPRESSION "ism_cooling.grackle_nT_table is not set")
set_tests_properties(ISMCoolingZMismatch PROPERTIES PASS_REGULAR_EXPRESSION "network.ism_metallicity = 0.5 but the table")
