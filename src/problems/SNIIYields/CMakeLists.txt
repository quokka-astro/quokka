if(AMReX_SPACEDIM EQUAL 3)
    set(JOB_NAME SNIIYields)

    quokka_add_problem(JOB_NAME ${JOB_NAME})
    add_test(
        NAME ${JOB_NAME}_zero_yield
        COMMAND ${JOB_NAME} ${CMAKE_SOURCE_DIR}/inputs/${JOB_NAME}.toml
            problem.test_zero_yield_only=1
            amrex.fpe_trap_invalid=1 amrex.fpe_trap_overflow=1 amrex.fpe_trap_zero=1)
    include(${CMAKE_SOURCE_DIR}/scripts/cmake/extract_chemical_yields.cmake)
endif()
