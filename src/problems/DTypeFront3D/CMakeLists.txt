# The problem is a spherical H II region around a central point source, so it is meaningful only in 3D.
if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization") # this will override
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

  set(JOB_NAME DTypeFront3D)

  # Number of species
  set(NSPEC 3)
  set(NEXTRASPEC 0)

  # Species enums
  set(SPECIES_ENUM
  "e = 0,
  H,
  Hp"
  )

  # Atomic masses (in atomic mass units)
  set(AION
  "5.4858e-4,
  1.0,
  1.0"
  )
  set(AION_INV
  "1822.89,
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
  set(NUM_CHEM_BANDS 1)
  set(CHEM_BANDS "3.29e15, 8.0e15") # Hz
  set(NUM_THERMAL_BANDS 2) # index 0 = IR, index 1 = optical, matching group_ir/group_optical in testDTypeFront3D.cpp

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
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY SKIP_PHOTOCHEMFLUX THERMAL_DUST_PHOTOCHEMISTRY)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
