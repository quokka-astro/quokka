if(AMReX_SPACEDIM EQUAL 3)
  # a GPU timing problem (128^3), too large for a CI test
  quokka_add_problem(JOB_NAME DTypeFront_GOW ADD_TEST OFF)
  # diagnostics: per-call burn statistics and a full dump of each failed burn (slows the kernel; not for timing)
  option(DTypeFront_GOW_BURN_STATS "DTypeFront_GOW: print burn statistics and failed-burn states" OFF)
  if(DTypeFront_GOW_BURN_STATS)
    target_compile_definitions(DTypeFront_GOW PRIVATE QUOKKA_BURN_STATS)
  endif()
  # the Rosenbrock tableau, for timing experiments: ROS2S, Rodas3P, Rodas4P or Rodas5P
  set(DTypeFront_GOW_TABLEAU "Rodas5P" CACHE STRING "DTypeFront_GOW: the Rosenbrock tableau")
  target_compile_definitions(DTypeFront_GOW PRIVATE QUOKKA_GOW_TABLEAU=${DTypeFront_GOW_TABLEAU})
  # partial pivoting in the LU factorization of the integrator matrix, for timing experiments
  option(DTypeFront_GOW_PIVOT "DTypeFront_GOW: pivot in the LU factorization" ON)
  if(NOT DTypeFront_GOW_PIVOT)
    target_compile_definitions(DTypeFront_GOW PRIVATE QUOKKA_GOW_NO_PIVOT)
  endif()
  # the network's shared evaluation of the right-hand side and the Jacobian (actual_rhs_and_jac), for timing experiments
  option(DTypeFront_GOW_SHARED_RHS_JAC "DTypeFront_GOW: evaluate the RHS and the Jacobian together" ON)
  if(NOT DTypeFront_GOW_SHARED_RHS_JAC)
    target_compile_definitions(DTypeFront_GOW PRIVATE QUOKKA_GOW_NO_SHARED_RHS_JAC)
  endif()
endif()
