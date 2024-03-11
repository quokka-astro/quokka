if (AMReX_SPACEDIM EQUAL 3)
	add_executable(first_star first_star.cpp ../fextract.cpp ${QuokkaObjSources})

	if(AMReX_GPU_BACKEND MATCHES "CUDA")
			setup_target_for_cuda_compilation(first_star)
	endif(AMReX_GPU_BACKEND MATCHES "CUDA")

	# Set string work_dir to ${CMAKE_SOURCE_DIR}/src/FirstStar/out/out-{YYMMDD-HHMMSS}
	# string(TIMESTAMP DATE_STR "%y%m%d-%H%M%S")
	# set(work_dir ${CMAKE_SOURCE_DIR}/src/FirstStar/out/out-${DATE_STR})
	set(work_dir ${PROJECT_SOURCE_DIR}/sims/240307-first-star/out)

	execute_process(
		COMMAND mkdir -p ${work_dir}
	)

	add_test(NAME FirstStar COMMAND first_star ./FirstStar.in WORKING_DIRECTORY ${work_dir})
endif()