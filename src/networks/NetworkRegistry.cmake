# Quokka-side microphysics network registry.

# ';' is written as '|' for AION_CONSTEXPR/ZION_CONSTEXPR because
# cmake_parse_arguments() splits single-value fields on ';'
# setup_target_for_microphysics_compilation() substitutes '|' back to ';'
set(_pi_hand_written_species_args
  NSPEC 3
  SPECIES_ENUM "H = 0, H_p, e"
  SPEC_NAMES "\"H\", \"H_p\", \"e\""
  SHORT_SPEC_NAMES "\"H\", \"H+\", \"e-\""
  AION "1.0, 1.0, 5.4858e-4"
  AION_INV "1.0, 1.0, 1822.89"
  ZION "1.0, 1.0, 1.0"
  # constexpr switch bodies
  AION_CONSTEXPR "case H:  a = 1.0| break| case H_p: a = 1.0| break| case e:  a = 1.0| break|"
  ZION_CONSTEXPR "case H:  z = 1.0| break| case H_p: z = 1.0| break| case e:  z = 1.0| break|"
  NUM_CHEM_BANDS 1
  CHEM_BANDS "13.6, 26.0" # eV
  POWER_LAW_INDEX 0 # jaff network.radiation.power_law_index
)

foreach(_pi_name photoionization_H_caseB photoionization_H_caseB_handwritten thermophotochemistry_handwritten)
  set(_pi_is_jaff FALSE)
  set(_pi_species_args "")
  set(_pi_jaff_toml_args "")
  set(_pi_num_thermal_bands "")
  if (_pi_name STREQUAL "photoionization_H_caseB")
    set(_pi_is_jaff TRUE)
    set(_pi_jaff_toml_args JAFF_TOML "${CMAKE_SOURCE_DIR}/src/networks/${_pi_name}/jaffgen.toml")
  elseif(_pi_name STREQUAL "photoionization_H_caseB_handwritten")
    set(_pi_species_args ${_pi_hand_written_species_args})
  elseif(_pi_name STREQUAL "thermophotochemistry_handwritten")
    set(_pi_species_args ${_pi_hand_written_species_args})
    set(_pi_num_thermal_bands 2)
  endif()

  register_microphysics_network(${_pi_name}
    EOSDIR            "photoionization"
    EOSPARAMFILE      "${CMAKE_SOURCE_DIR}/extern/Microphysics/EOS/photoionization/_parameters"
    # No PARAMFILE: only read when UNIT_TEST_GATE is ON, and no problem in this
    # family ships a unit-test _parameters file.
    NETWORKPARAMFILE  "${CMAKE_SOURCE_DIR}/src/networks/${_pi_name}/_parameters"
    HAS_NET_FILE      FALSE
    HEADER_TEMPLATE   "${CMAKE_SOURCE_DIR}/src/networks/network_header.template"
    EXTRA_SOURCES     "${CMAKE_SOURCE_DIR}/extern/Microphysics/interfaces/eos_data.cpp"
                      "${CMAKE_SOURCE_DIR}/extern/Microphysics/interfaces/network_initialization.cpp"
                      "${CMAKE_SOURCE_DIR}/extern/Microphysics/EOS/photoionization/actual_eos_data.cpp"
                      "${CMAKE_SOURCE_DIR}/src/networks/${_pi_name}/actual_network_data.cpp"
    UNIT_TEST_GATE    "BUILD_UNIT_TEST_PC"
    IS_JAFF           ${_pi_is_jaff}
    USES_INTEGRATOR_DIRS TRUE
    NETWORK_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/src/networks/${_pi_name}"
                          "${CMAKE_SOURCE_DIR}/extern/Microphysics/networks"
    NUM_THERMAL_BANDS ${_pi_num_thermal_bands}
    ${_pi_species_args}
    ${_pi_jaff_toml_args}
  )
endforeach()
