if(AMReX_SPACEDIM EQUAL 3)
    set(JOB_NAME WRAGBYields)

    quokka_add_problem(JOB_NAME ${JOB_NAME})
    include(${CMAKE_SOURCE_DIR}/scripts/cmake/extract_chemical_yields.cmake)

endif()
