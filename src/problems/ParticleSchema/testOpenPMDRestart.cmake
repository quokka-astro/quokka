# Integration: AMR + tracers, native compatibility, and continued field equality.
# Start clean: a stale native checkpoint with an openPMD name would take precedence on restart.
file(REMOVE_RECURSE "${RUN_DIR}")
file(MAKE_DIRECTORY "${RUN_DIR}")
set(common "${INPUT}" "amr.n_cell=16 16 16" amr.max_grid_size=8
  amr.blocking_factor=8 amr.max_level=1 max_timesteps=2 checkpoint_interval=1
  plotfile_interval=-1 quokka.slice_z.int=-1 blast_problem.check_solution=0)
function(run_checked label)
  execute_process(COMMAND ${ARGN} WORKING_DIRECTORY "${RUN_DIR}"
    RESULT_VARIABLE result OUTPUT_FILE "${RUN_DIR}/${label}.log" ERROR_FILE "${RUN_DIR}/${label}.err")
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${label} failed (${result}); see ${RUN_DIR}/${label}.log and .err")
  endif()
endfunction()
run_checked(reference "${SIM}" ${common} checkpoint_format=amrex checkpoint_prefix=reference)
run_checked(openpmd "${SIM}" ${common} checkpoint_format=openpmd checkpoint_prefix=openpmd)
set(restart_command "${SIM}")
if(MPIEXEC)
  set(restart_command "${MPIEXEC}" "${NUMPROC_FLAG}" 2 "${SIM}")
endif()
run_checked(restart ${restart_command} ${common} restartfile=openpmd0000001 checkpoint_prefix=restarted)
run_checked(compare "${COMPARE}" reference=reference0000002 candidate=restarted0000002 levels=2)
run_checked(native_restart "${SIM}" ${common} restartfile=reference0000001 checkpoint_prefix=native_restarted)
run_checked(native_compare "${COMPARE}" reference=reference0000002 candidate=native_restarted0000002 levels=2)
run_checked(refined_openpmd "${SIM}" ${common} "amr.n_cell=32 32 32"
  restartfile=openpmd0000001 checkpoint_prefix=refined_openpmd)
run_checked(refined_native "${SIM}" ${common} "amr.n_cell=32 32 32"
  restartfile=reference0000001 checkpoint_prefix=refined_native)
run_checked(refined_compare "${COMPARE}" reference=refined_native0000002 candidate=refined_openpmd0000002 levels=2)
