if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(
    JOB_NAME MolecularCloudDisruption
    INPUT_FILE MolecularCloudDisruption_regression.toml
    ADD_TEST ON
    PRIORITY 10)

  add_test(
    NAME MolecularCloudDisruptionRestart
    COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:MolecularCloudDisruption>
            -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/restart_test
            -DENABLE_TESTS_FPE=${ENABLE_TESTS_FPE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/testRestart.cmake)
endif()
