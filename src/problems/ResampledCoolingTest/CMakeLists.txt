quokka_add_problem(JOB_NAME ResampledCoolingTest)
add_test(NAME ResampledCoolingTestV2 COMMAND ResampledCoolingTest ../inputs/ResampledCoolingTestV2.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
