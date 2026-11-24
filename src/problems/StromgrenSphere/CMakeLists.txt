if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME StromgrenSphere)
  quokka_add_problem(JOB_NAME ${JOB_NAME})
  set(STROMGREN_C_HAT_DIVISOR 1 CACHE STRING "StromgrenSphere reduced speed of light: c_hat = c / divisor")
  target_compile_definitions(${JOB_NAME} PRIVATE C_HAT_DIVISOR=${STROMGREN_C_HAT_DIVISOR})
  add_test(NAME ${JOB_NAME}TempIndependentRecombination COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}TempIndependentRecombination.toml ${QuokkaTestParams}
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
