# Quokka-side microphysics network registry.

foreach(_pi_name photoionization dtype_jaff)
  # photoionization itself is hand-written; the other two are jaff-generated.
  set(_pi_is_jaff FALSE)
  if (_pi_name STREQUAL "dtype_jaff")
    set(_pi_is_jaff TRUE)
  endif()

  register_microphysics_network(${_pi_name}
    EOSDIR            "photoionization"
    EOSPARAMFILE      "${CMAKE_SOURCE_DIR}/extern/Microphysics/EOS/photoionization/_parameters"
    # No PARAMFILE: only read when UNIT_TEST_GATE is ON, and no problem in this
    # family ships a unit-test _parameters file.
    NETWORKPARAMFILE  "${CMAKE_SOURCE_DIR}/src/networks/${_pi_name}/_parameters"
    HAS_NET_FILE      FALSE
    EXTRA_SOURCES     "${CMAKE_SOURCE_DIR}/extern/Microphysics/interfaces/eos_data.cpp"
                      "${CMAKE_SOURCE_DIR}/extern/Microphysics/interfaces/network_initialization.cpp"
                      "${CMAKE_SOURCE_DIR}/extern/Microphysics/EOS/photoionization/actual_eos_data.cpp"
                      "${CMAKE_SOURCE_DIR}/src/networks/${_pi_name}/actual_network_data.cpp"
    UNIT_TEST_GATE    "BUILD_UNIT_TEST_PC"
    IS_JAFF           ${_pi_is_jaff}
    USES_INTEGRATOR_DIRS TRUE
    NETWORK_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/src/networks/${_pi_name}"
                          "${CMAKE_SOURCE_DIR}/extern/Microphysics/networks"
  )
endforeach()
