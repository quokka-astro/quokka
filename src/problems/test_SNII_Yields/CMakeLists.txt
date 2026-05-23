if(AMReX_SPACEDIM EQUAL 3)
	set(JOB_NAME test_SNII_Yields)

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
			-DQUOKKA_YIELD_ARCHIVE=${CMAKE_SOURCE_DIR}/extern/yields/yields.tar.gz
			-DQUOKKA_YIELD_DESTINATION=${CMAKE_SOURCE_DIR}/extern/yields
			-DQUOKKA_YIELD_SENTINEL=${CMAKE_SOURCE_DIR}/extern/yields/SNII_Sukhbold16
			-P ${CMAKE_SOURCE_DIR}/scripts/cmake/extract_chemical_yields.cmake)
	add_test(
		NAME ${JOB_NAME}_validate
		COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/validate_chemical_yields.py
			--case snii
			--plotdir ${CMAKE_SOURCE_DIR}/tests/${JOB_NAME}
			--yield-root ${CMAKE_SOURCE_DIR}/extern/yields
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
	set_tests_properties(${JOB_NAME}_prepare_yields PROPERTIES FIXTURES_SETUP ${JOB_NAME}_yields_fixture)
	set_tests_properties(${JOB_NAME} PROPERTIES FIXTURES_REQUIRED ${JOB_NAME}_yields_fixture FIXTURES_SETUP ${JOB_NAME}_fixture)
	set_tests_properties(${JOB_NAME}_validate PROPERTIES FIXTURES_REQUIRED ${JOB_NAME}_fixture)
endif()
