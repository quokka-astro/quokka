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

  set(JOB_NAME DTypeFrontVC)

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

  set(SHORT_SPEC_NAMES
  "\"H\",
   \"H+\",
   \"e-\""
  )

  # Atomic masses (in atomic mass units) and charges. EOS/photoionization
  # computes everything it needs from spmasses/gammas directly, but eos() in
  # interfaces/eos.H unconditionally calls composition(), which forms
  # sum(xn[n] * aion_inv[n]) and inverts it -- leaving these unset makes
  # configure_file() emit empty (i.e. all-zero) arrays and that inversion
  # divides by zero.
  set(AION
  "1.0,
   1.0,
   5.4858e-4"
  )
  set(AION_INV
  "1.0,
   1.0,
   1822.89"
  )

  set(ZION
  "1.0,
   1.0,
   1.0"
  )

  # constexpr switch bodies
  set(AION_CONSTEXPR
  "case H:  a = 1.0; break;
   case Hp: a = 1.0; break;
   case e:  a = 1.0; break;"
  )

  set(ZION_CONSTEXPR
  "case H:  z = 1.0; break;
   case Hp: z = 1.0; break;
   case e:  z = 1.0; break;"
  )

  # No extra network properties
  set(PROPERTIES "")

  # Set photoionisation properties
  set(NUM_CHEM_BANDS 1)
  set(CHEM_BANDS "13.6, 62.1") # eV
  set(POWER_LAW_INDEX 0) # jaff network.radiation.power_law_index

  # Configure the header
  configure_file(
      ${CMAKE_SOURCE_DIR}/src/networks/network_header.template
      ${CMAKE_CURRENT_BINARY_DIR}/network_properties.H
      @ONLY
  )


  add_executable(
    ${JOB_NAME} testDTypeFrontVC.cpp "${QuokkaSourcesNoEOS}"
                ../../radiation/photochemistry.cpp ${photoionization_sources})

  # this will add #define CHEMISTRY
  target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME DTypeFrontVC COMMAND ${JOB_NAME} ../inputs/DTypeFrontVC.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()