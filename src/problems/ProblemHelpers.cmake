# Helper function to set up a Quokka problem executable and optionally its test
# Usage:
#   quokka_add_problem(
#     JOB_NAME <name>
#     [PRIORITY <0-9>]           # default: 9, lower runs first
#     [INPUT_FILE <input_file>]  # default: ${JOB_NAME}.in
#     [ADD_TEST <ON|OFF>]        # default: ON
#     [TEST_PARAMS <params>]     # default: ${QuokkaTestParams}
#   )
#
# Tests are registered and added later via quokka_add_registered_tests(),
# sorted by priority (ascending) then by name (alphabetical).
function(quokka_add_problem)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "JOB_NAME;PRIORITY;INPUT_FILE;ADD_TEST;TEST_PARAMS" "")

  if(NOT ARG_JOB_NAME)
    message(FATAL_ERROR "quokka_add_problem: JOB_NAME is required")
  endif()

  # Defaults
  if(NOT DEFINED ARG_PRIORITY)
    set(ARG_PRIORITY 9)
  endif()
  if(NOT ARG_INPUT_FILE)
    set(ARG_INPUT_FILE "${ARG_JOB_NAME}.in")
  endif()
  if(NOT ARG_TEST_PARAMS)
    set(ARG_TEST_PARAMS "${QuokkaTestParams}")
  endif()

  # Determine if test should be added
  set(_add_test TRUE)
  if(DEFINED ARG_ADD_TEST AND ARG_ADD_TEST STREQUAL "OFF")
    set(_add_test FALSE)
  endif()

  # Add executable
  add_executable(${ARG_JOB_NAME} test${ARG_JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${ARG_JOB_NAME})
  endif()

  # Register test for later sorted addition
  if(_add_test)
    set_property(GLOBAL APPEND PROPERTY QUOKKA_TEST_KEYS "${ARG_PRIORITY}_${ARG_JOB_NAME}")
    set_property(GLOBAL PROPERTY QUOKKA_TEST_${ARG_JOB_NAME}_INPUT "${ARG_INPUT_FILE}")
    set_property(GLOBAL PROPERTY QUOKKA_TEST_${ARG_JOB_NAME}_PARAMS "${ARG_TEST_PARAMS}")
  endif()
endfunction()

# Add all registered tests sorted by priority then name
function(quokka_add_registered_tests)
  get_property(_keys GLOBAL PROPERTY QUOKKA_TEST_KEYS)
  if(NOT _keys)
    return()
  endif()

  list(SORT _keys)

  foreach(_key ${_keys})
    string(REGEX REPLACE "^[0-9]_" "" _name "${_key}")
    get_property(_input GLOBAL PROPERTY QUOKKA_TEST_${_name}_INPUT)
    get_property(_params GLOBAL PROPERTY QUOKKA_TEST_${_name}_PARAMS)

    add_test(NAME ${_name}
      COMMAND ${_name} ../inputs/${_input} ${_params}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  endforeach()
endfunction()