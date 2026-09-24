file(MAKE_DIRECTORY "${WORK_DIR}")

set(common_args "${INPUT_FILE}" "max_timesteps=0" "plotfile_interval=-1" "checkpoint_interval=-1")

execute_process(
  COMMAND "${HYDRO_CONTACT_EXE}" ${common_args} "heating_rate_external=yr+kyr+Myr+Gyr"
  WORKING_DIRECTORY "${WORK_DIR}"
  RESULT_VARIABLE known_result
  OUTPUT_VARIABLE known_output
  ERROR_VARIABLE known_error)
if(NOT known_result STREQUAL "0")
  message(FATAL_ERROR "Known time units failed (${known_result}):\n${known_output}\n${known_error}")
endif()

execute_process(
  COMMAND "${HYDRO_CONTACT_EXE}" ${common_args} "heating_rate_external=not_a_unit"
  WORKING_DIRECTORY "${WORK_DIR}"
  RESULT_VARIABLE unknown_result
  OUTPUT_VARIABLE unknown_output
  ERROR_VARIABLE unknown_error)
if(unknown_result STREQUAL "0" OR NOT "${unknown_output}${unknown_error}" MATCHES "Unknown variable not_a_unit")
  message(FATAL_ERROR "Unknown unit was not rejected as expected (${unknown_result}):\n${unknown_output}\n${unknown_error}")
endif()
