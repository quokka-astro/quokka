quokka_add_problem(JOB_NAME YieldValidation ADD_TEST OFF)
add_test(NAME YieldValidation COMMAND YieldValidation
    amrex.fpe_trap_invalid=1 amrex.fpe_trap_overflow=1 amrex.fpe_trap_zero=1)
