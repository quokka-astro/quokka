set(microphysics_network_name "primordial_chem") # this will override
                                                 # network_name to
                                                 # primordial_chem for this
                                                 # directory only
setup_target_for_microphysics_compilation(${microphysics_network_name}
                                          "${CMAKE_CURRENT_BINARY_DIR}/")

# use the BEFORE keyword so that these files get priority in compilation for
# targets in this directory this is critical to ensure the correct Microphysics
# files are linked to primordial chem target
include_directories(
  BEFORE ${primordial_chem_dirs} "${CMAKE_CURRENT_BINARY_DIR}/"
  "includes/extern_parameters.H" "includes/network_properties.H")

set(JOB_NAME PrimordialChem)

add_executable(
  ${JOB_NAME} testPrimordialChem.cpp "${QuokkaSourcesNoEOS}"
              ${primordial_chem_sources}
              ${CMAKE_SOURCE_DIR}/src/networks/primordial_chem/network.cpp)

# the network's actual_rhs.H, ahead of Microphysics' copy
target_include_directories(${JOB_NAME} BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/src/networks/primordial_chem)

# this will add #define CHEMISTRY
target_compile_definitions(${JOB_NAME} PUBLIC CHEMISTRY)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
