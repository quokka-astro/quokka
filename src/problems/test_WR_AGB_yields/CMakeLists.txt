if(AMReX_SPACEDIM EQUAL 3)
    set(JOB_NAME test_WR_AGB_yields)

    add_executable(${JOB_NAME} ${JOB_NAME}.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(${JOB_NAME})
    endif()

    add_test(
        NAME ${JOB_NAME}
        COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(
        NAME ${JOB_NAME}_prepare_yields
        COMMAND ${CMAKE_COMMAND}
            -DQUOKKA_YIELD_ARCHIVE=${CMAKE_SOURCE_DIR}/extern/yields/quokka_yield_tables.tar.gz
            -DQUOKKA_YIELD_DESTINATION=${CMAKE_SOURCE_DIR}/extern/yields
            -DQUOKKA_YIELD_SENTINEL=${CMAKE_SOURCE_DIR}/extern/yields/WR_mass_loss_distribution_table.csv
            -P ${CMAKE_SOURCE_DIR}/scripts/cmake/extract_chemical_yields.cmake)
    set_tests_properties(${JOB_NAME}_prepare_yields PROPERTIES FIXTURES_SETUP ${JOB_NAME}_yields_fixture)
    set_tests_properties(${JOB_NAME} PROPERTIES FIXTURES_REQUIRED ${JOB_NAME}_yields_fixture)

endif()
