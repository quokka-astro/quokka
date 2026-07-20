if(AMReX_SPACEDIM EQUAL 3)
	set(JOB_NAME ParticleSF)

	add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
	if(AMReX_GPU_BACKEND MATCHES "CUDA")
		setup_target_for_cuda_compilation(${JOB_NAME})
	endif()

  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}2 COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}2.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}_restart_cap COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}_restart_cap.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}_restart_cap2 COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}_restart_cap2.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

	set_tests_properties(${JOB_NAME} PROPERTIES COST 100)
	set_tests_properties(${JOB_NAME}2 PROPERTIES DEPENDS "${JOB_NAME}" COST 99)
	set_tests_properties(${JOB_NAME}_restart_cap PROPERTIES COST 99)
	set_tests_properties(${JOB_NAME}_restart_cap2 PROPERTIES DEPENDS "${JOB_NAME}_restart_cap" COST 98)
endif()
