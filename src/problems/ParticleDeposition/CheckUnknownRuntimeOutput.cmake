execute_process(
  COMMAND "${QUOKKA_TEST_BINARY}" "${QUOKKA_TEST_INPUT}" "derived_vars=${QUOKKA_TEST_OUTPUT_NAME}"
  WORKING_DIRECTORY "${QUOKKA_TEST_WORKDIR}"
  RESULT_VARIABLE test_result
  OUTPUT_VARIABLE test_stdout
  ERROR_VARIABLE test_stderr
)

if("${test_result}" STREQUAL "0")
  message(FATAL_ERROR "Expected an abort for unknown runtime-derived output '${QUOKKA_TEST_OUTPUT_NAME}', but the run succeeded.")
endif()

set(test_output "${test_stdout}")
string(APPEND test_output "${test_stderr}")

if(NOT test_output MATCHES "Requested runtime derived field output '${QUOKKA_TEST_OUTPUT_NAME}' is not emitted by any configured provider")
  message(FATAL_ERROR "Run failed, but not with the expected runtime-derived output validation message.\n${test_output}")
endif()
