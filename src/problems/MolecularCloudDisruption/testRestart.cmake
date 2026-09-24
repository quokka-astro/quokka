# Use a unique directory so repeated or concurrent runs cannot reuse old checkpoints.
string(RANDOM LENGTH 12 run_id)
set(run_dir "${TEST_DIR}/${run_id}")
file(MAKE_DIRECTORY "${run_dir}")

set(common_args
    "${SOURCE_DIR}/inputs/MolecularCloudDisruption_regression.toml"
    "problem.stellar_particles_file=${SOURCE_DIR}/inputs/MolecularCloudDisruption_particles.txt"
    "cooling.hdf5_data_file=\"${SOURCE_DIR}/extern/cooling/CloudyData_UVB=HM2012_shielded_resampled_noPE.h5\""
    "statistics_interval=1")
if(ENABLE_TESTS_FPE)
  list(APPEND common_args amrex.fpe_trap_invalid=1 amrex.fpe_trap_zero=1 amrex.fpe_trap_overflow=1)
endif()

execute_process(
  COMMAND "${EXECUTABLE}" ${common_args} checkpoint_interval=1 checkpoint_prefix=restart_chk statistics_file=fresh_history.txt
  WORKING_DIRECTORY "${run_dir}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Fresh start failed: ${output}\n${error}")
endif()

# Different reference masses exercise both initial-mass guards without requiring
# a long evolution to form stars and expel cloud gas.
execute_process(
  COMMAND "${EXECUTABLE}" ${common_args} restartfile=restart_chk0000001 max_timesteps=2
          problem.stellar_mass_Msun=1000 problem.cloud_mass_Msun=200000 statistics_file=restart_history.txt
  WORKING_DIRECTORY "${run_dir}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Checkpoint restart failed: ${output}\n${error}")
endif()

file(STRINGS "${run_dir}/restart_history.txt" history)
set(rows 0)
foreach(line IN LISTS history)
  if(line MATCHES "^# cycle ")
    string(REGEX REPLACE "^# " "" header "${line}")
    separate_arguments(columns UNIX_COMMAND "${header}")
    list(FIND columns "t_over_tff" tff_column)
    if(tff_column LESS 0)
      message(FATAL_ERROR "Missing t_over_tff diagnostic")
    endif()
  elseif(NOT line MATCHES "^#")
    separate_arguments(values UNIX_COMMAND "${line}")
    list(GET values ${tff_column} t_over_tff)
    if(NOT t_over_tff MATCHES "^[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$" OR t_over_tff LESS_EQUAL 0)
      message(FATAL_ERROR "Invalid restart free-fall normalization: ${t_over_tff}")
    endif()
    math(EXPR rows "${rows} + 1")
  endif()
endforeach()
if(rows LESS 2)
  message(FATAL_ERROR "Expected statistics at checkpoint loading and after restart evolution")
endif()
