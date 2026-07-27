set(microphysics_network_name "photoionization_jaff") # this will override
                                                 # network_name to
                                                 # photoionization for this
                                                 # directory only
setup_target_for_microphysics_compilation(${microphysics_network_name}
                                          "${CMAKE_CURRENT_BINARY_DIR}/"
                                          "Rosenbrock")

# use the BEFORE keyword so that these files get priority in compilation for
# targets in this directory this is critical to ensure the correct Microphysics
# files are linked to photoionization target
include_directories(
  BEFORE ${photoionization_dirs} "${CMAKE_CURRENT_BINARY_DIR}/"
  "includes/extern_parameters.H" "includes/network_properties.H")

set(JOB_NAME OneZonePhotoionizationJaff)

# NSPEC/SPECIES_ENUM/SPEC_NAMES/NUM_CHEM_BANDS/CHEM_BANDS are derived from
# jaff's own _parameters and jaff.toml by write_jaff_network_vars.py (invoked
# from setup_target_for_microphysics_compilation above), instead of being
# hand-duplicated here.
include(${jaff_network_vars_file})
set(NEXTRASPEC 0)

# AION/AION_INV/ZION/AION_CONSTEXPR/ZION_CONSTEXPR/SHORT_SPEC_NAMES are unused
# for this network: EOS/photoionization/actual_eos.H computes everything from
# spmasses/gammas directly and never calls the generic composition() helpers
# in eos_composition.H that read aion[]/zion[], and short_spec_names_cxx is
# only read by unit_test/ debug utilities not built for this target.

# No extra network properties
set(PROPERTIES "")

# Configure the header
configure_file(
    ${CMAKE_SOURCE_DIR}/src/networks/network_header.template
    ${CMAKE_CURRENT_BINARY_DIR}/network_properties.H
    @ONLY
)


add_executable(
  ${JOB_NAME} testOneZonePhotoionization.cpp "${QuokkaSourcesNoEOS}"
              ../../radiation/photochemistry.cpp ${photoionization_sources})

# this will add #define PHOTOCHEMISTRY
target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME OneZonePhotoionizationNoEnergy COMMAND ${JOB_NAME} ../inputs/OneZonePhotoionization.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

