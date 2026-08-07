if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization") # this will override
                                                  # network_name to
                                                  # photoionization for this
                                                  # directory only
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/"
                                            INTEGRATOR "Rosenbrock")

  # use the BEFORE keyword so that these files get priority in compilation for
  # targets in this directory this is critical to ensure the correct Microphysics
  # files are linked to photoionization target
  include_directories(
    BEFORE ${photoionization_dirs} "${CMAKE_CURRENT_BINARY_DIR}/"
    "includes/extern_parameters.H" "includes/network_properties.H")

  set(JOB_NAME StromgrenSphere)

  # Number of species
  set(NSPEC 3)
  set(NEXTRASPEC 0)

  # Species enums
  set(SPECIES_ENUM
  "H = 0,
  Hp,
  e"
  )

  # Species names
  set(SPEC_NAMES
  "\"H\",
  \"Hp\",
  \"e\""
  )

  # No extra network properties
  set(PROPERTIES "")

  # Set photoionisation properties
  set(NUM_CHEM_BANDS 1)
  set(CHEM_BANDS "13.6, 62.1") # eV

  # Configure the header
  configure_file(
      ${CMAKE_SOURCE_DIR}/src/networks/network_header.template
      ${CMAKE_CURRENT_BINARY_DIR}/network_properties.H
      @ONLY
  )


  add_executable(
    ${JOB_NAME} test${JOB_NAME}.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${photoionization_sources})

  # this will add #define PHOTOCHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}TempIndependentRecombination COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}TempIndependentRecombination.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()