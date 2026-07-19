# Helper function to set up a Quokka problem executable and optionally its test
# Usage:
#   quokka_add_problem(
#     JOB_NAME <name>
#     [INPUT_FILE <input_file>]  # defaults to ${JOB_NAME}.toml
#     [ADD_TEST <ON|OFF>]        # whether to add a test (default: ON)
#     [TEST_PARAMS <params>]     # additional test parameters (default: ${QuokkaTestParams})
#     [PRIORITY <number>]        # test priority, higher values run first (default: 0)
#   )
function(quokka_add_problem)
  cmake_parse_arguments(
    QUOKKA_PROBLEM
    ""
    "JOB_NAME;INPUT_FILE;ADD_TEST;TEST_PARAMS;PRIORITY"
    ""
    ${ARGN}
  )

  if(NOT QUOKKA_PROBLEM_JOB_NAME)
    message(FATAL_ERROR "quokka_add_problem: JOB_NAME is required")
  endif()

  # Set default input file if not provided
  if(NOT QUOKKA_PROBLEM_INPUT_FILE)
    set(QUOKKA_PROBLEM_INPUT_FILE "${QUOKKA_PROBLEM_JOB_NAME}.toml")
  endif()

  # Default to adding test if not explicitly set, or if explicitly set to ON
  if(NOT DEFINED QUOKKA_PROBLEM_ADD_TEST)
    set(_add_test TRUE)
  elseif(QUOKKA_PROBLEM_ADD_TEST STREQUAL "ON")
    set(_add_test TRUE)
  elseif(QUOKKA_PROBLEM_ADD_TEST STREQUAL "OFF")
    set(_add_test FALSE)
  else()
    message(FATAL_ERROR "quokka_add_problem: ADD_TEST must be ON or OFF, got '${QUOKKA_PROBLEM_ADD_TEST}'")
  endif()

  # Set default test params
  if(NOT QUOKKA_PROBLEM_TEST_PARAMS)
    set(QUOKKA_PROBLEM_TEST_PARAMS "${QuokkaTestParams}")
  endif()

  # Set default priority (0 = lowest priority, higher values = higher priority)
  if(NOT DEFINED QUOKKA_PROBLEM_PRIORITY)
    set(QUOKKA_PROBLEM_PRIORITY 0)
  endif()

  # Add executable
  add_executable(${QUOKKA_PROBLEM_JOB_NAME} 
    test${QUOKKA_PROBLEM_JOB_NAME}.cpp ${QuokkaObjSources})

  # Setup CUDA compilation if needed
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${QUOKKA_PROBLEM_JOB_NAME})
  endif()

  # Add test if requested
  if(_add_test)
    add_test(
      NAME ${QUOKKA_PROBLEM_JOB_NAME}
      COMMAND ${QUOKKA_PROBLEM_JOB_NAME} ../inputs/${QUOKKA_PROBLEM_INPUT_FILE} ${QUOKKA_PROBLEM_TEST_PARAMS}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # Set test priority using COST property (higher COST = runs first)
    set_tests_properties(${QUOKKA_PROBLEM_JOB_NAME} PROPERTIES COST ${QUOKKA_PROBLEM_PRIORITY})
  endif()
endfunction()