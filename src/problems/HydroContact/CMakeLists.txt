quokka_add_problem(JOB_NAME HydroContact)
if(BUILD_TESTING)
  add_test(
    NAME ParmParseUnitSuffixes
    COMMAND ${CMAKE_COMMAND}
      -DHYDRO_CONTACT_EXE=$<TARGET_FILE:HydroContact>
      -DINPUT_FILE=${QuokkaCode_SOURCE_DIR}/inputs/HydroContact.toml
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/ParmParseUnitSuffixes
      -P ${QuokkaCode_SOURCE_DIR}/src/util/tests/check_parmparse_unit_suffixes.cmake)
endif()
