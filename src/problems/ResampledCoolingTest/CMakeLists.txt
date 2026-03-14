quokka_add_problem(JOB_NAME ResampledCoolingTest)
if(AMReX_SPACEDIM EQUAL 3)
	add_test(NAME ResampledCoolingTestV2 COMMAND ResampledCoolingTest ../inputs/ResampledCoolingTestV2.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
