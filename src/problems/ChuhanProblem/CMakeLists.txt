if(AMReX_SPACEDIM EQUAL 3)
	set(JOB_NAME ChuhanProblem)

	add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
	if(AMReX_GPU_BACKEND MATCHES "CUDA")
		setup_target_for_cuda_compilation(${JOB_NAME})
	endif()

	add_test(
		NAME ${JOB_NAME}
		COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

	# Post-processing target: Generate metallicity plots for all timesteps
	add_custom_target(${JOB_NAME}_plots
		COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/python/plot_chuhan_metallicity.py
			${CMAKE_SOURCE_DIR}/tests/chuhan_*/plt*/
			-o ${CMAKE_SOURCE_DIR}/tests/chuhan_plots
			-j 4
		COMMENT "Generating metallicity (chemical scalar) plots for ${JOB_NAME}..."
		VERBATIM
	)

	# Make plots target depend on test completion
	add_dependencies(${JOB_NAME}_plots ${JOB_NAME})
endif()
