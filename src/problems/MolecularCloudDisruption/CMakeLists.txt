if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(
    JOB_NAME MolecularCloudDisruption
    INPUT_FILE MolecularCloudDisruption_regression.toml
    ADD_TEST ON
    PRIORITY 10)
endif()
