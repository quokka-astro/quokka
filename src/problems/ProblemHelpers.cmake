# Helper function to set up a Quokka problem executable and optionally its test
# Usage:
#   quokka_add_problem(
#     JOB_NAME <name>
#     [PRIORITY <0-9>]           # default: 9, lower runs first (uses CTest COST property)
#     [INPUT_FILE <input_file>]  # default: ${JOB_NAME}.in
#     [ADD_TEST <ON|OFF>]        # default: ON
#     [TEST_PARAMS <params>]     # default: ${QuokkaTestParams}
#   )
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

  # Add test with priority using CTest COST property
  if(_add_test)
    add_test(NAME ${ARG_JOB_NAME}
      COMMAND ${ARG_JOB_NAME} ../inputs/${ARG_INPUT_FILE} ${ARG_TEST_PARAMS}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(${ARG_JOB_NAME} PROPERTIES COST ${ARG_PRIORITY})
  endif()
endfunction()