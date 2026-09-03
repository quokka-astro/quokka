if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization")
  set(DiskGalaxy_INTEGRATOR "Rosenbrock" CACHE STRING
      "Microphysics integrator for DiskGalaxy")
  set_property(CACHE DiskGalaxy_INTEGRATOR PROPERTY STRINGS VODE Rosenbrock)
  setup_target_for_microphysics_compilation(
    ${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/"
    "${DiskGalaxy_INTEGRATOR}")

  # These generated/network headers must take precedence over the gamma-law
  # headers configured in src/CMakeLists.txt.
  include_directories(
    BEFORE ${photoionization_dirs} "${CMAKE_CURRENT_BINARY_DIR}/"
    "includes/extern_parameters.H" "includes/network_properties.H")

  set(NSPEC 3)
  set(NEXTRASPEC 0)

  set(SPECIES_ENUM
  "e = 0,
  H,
  Hp"
  )

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
  set(ZION
  "1.0,
  1.0,
  1.0"
  )

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

  set(PROPERTIES "")
  set(NUM_CHEM_BANDS 1)
  set(CHEM_BANDS "3.29e15, 1.50e16") # Hz

  configure_file(
    ${CMAKE_SOURCE_DIR}/src/networks/network_header.template
    ${CMAKE_CURRENT_BINARY_DIR}/network_properties.H @ONLY)

  add_executable(
    DiskGalaxy testDiskGalaxy.cpp "${QuokkaSourcesNoEOS}"
    ../../radiation/photochemistry.cpp ${photoionization_sources})
  target_compile_definitions(DiskGalaxy PUBLIC PHOTOCHEMISTRY)

  if(DiskGalaxy_INTEGRATOR STREQUAL "Rosenbrock")
    target_compile_definitions(DiskGalaxy PUBLIC DISKGALAXY_USE_ROSENBROCK)
  else()
    target_compile_definitions(DiskGalaxy PUBLIC DISKGALAXY_USE_VODE)
  endif()

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(DiskGalaxy)
  endif()

  # extract data files for regression testing
  file(
    ARCHIVE_EXTRACT INPUT
    ${QuokkaCode_SOURCE_DIR}/extern/agora_data/AgoraGalaxy_particles_LOW.tar.gz
    DESTINATION ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
  file(ARCHIVE_EXTRACT INPUT
       ${QuokkaCode_SOURCE_DIR}/extern/agora_data/vcirc.tar.gz DESTINATION
       ${QuokkaCode_SOURCE_DIR}/extern/agora_data)
endif()
