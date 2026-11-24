if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME StromgrenSphereRSLA)

  set(STROMGREN_RSLA_C_HAT_DIVISOR 100 CACHE STRING "StromgrenSphereRSLA reduced speed of light: c_hat = c / divisor")
  target_compile_definitions(StromgrenSphereRSLA PRIVATE C_HAT_DIVISOR=${STROMGREN_RSLA_C_HAT_DIVISOR})
endif()
