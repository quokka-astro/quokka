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
