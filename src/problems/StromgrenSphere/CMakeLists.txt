if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME StromgrenSphere)
  quokka_add_problem(JOB_NAME ${JOB_NAME})
  add_test(NAME ${JOB_NAME}TempIndependentRecombination COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}TempIndependentRecombination.toml ${QuokkaTestParams}
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
