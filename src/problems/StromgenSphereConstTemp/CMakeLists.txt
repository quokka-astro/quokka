if(AMReX_SPACEDIM GREATER_EQUAL 2)
  set(microphysics_network_name "H_ionization_TInd") # this will override
                                                  # network_name to
                                                  # photoionization for this
                                                  # directory only
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/")

  # use the BEFORE keyword so that these files get priority in compilation for
  # targets in this directory this is critical to ensure the correct Microphysics
  # files are linked to photoionization target
  include_directories(
    BEFORE ${photoionization_dirs} "${CMAKE_CURRENT_BINARY_DIR}/"
    "includes/extern_parameters.H" "includes/network_properties.H")

  set(JOB_NAME StromgenSphereConstTemp)

  # Number of species
  set(NSPEC 3)
  # set(NSPEC 2)
  set(NEXTRASPEC 0)

  # Species enums
  set(SPECIES_ENUM
  "e = 0,
  H,
  Hp"
  )

  # Atomic masses (in atomic mass units)
  set(AION
  "1.0,
  1.0,
  1.0"
  )
  set(AION_INV
  "1.0,
  1.0,
  1.0"
  )

  # Charges
  set(ZION
  "1.0,
  1.0,
  1.0"
  )

  # constexpr switch bodies
  set(AION_CONSTEXPR
  "case e:  a = 1.0; break;
  case H:  a = 1.0; break;
  case Hp: a = 1.0; break;"
  )

  set(ZION_CONSTEXPR
  "case e:  z = 1.0; break;
  case H:  z = 1.0; break;
  case Hp: z = 1.0; break;"
  )

  # Species names
  set(SHORT_SPEC_NAMES
  "\"e\",
  \"H\",
  \"H+\""
  )

  set(SPEC_NAMES
  "\"Electron\",
  \"Hydrogen\",
  \"Hydrogen_Ion\""
  )

  # No extra network properties
  set(PROPERTIES "")

  # Set photoionisation properties
  set(NUM_CHEM_ACTIVE_RAD_GROUPS 1)
  set(CHEM_ACTIVE_RAD_FREQ_BOUNDS "3.29e15, 1.50e16") # Hz
  # set(CHEM_ACTIVE_RAD_FREQ_BOUNDS "1.0, 5.0") # Hz

  # Configure the header
  configure_file(
      ${CMAKE_SOURCE_DIR}/src/networks/network_header.template
      ${CMAKE_CURRENT_BINARY_DIR}/network_properties.H
      @ONLY
  )


  add_executable(
    ${JOB_NAME} testStromgenSphereConstTemp.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${photoionization_sources})

  # this will add #define CHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(
    NAME ${JOB_NAME}
    COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in ${QuokkaTestParams}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()