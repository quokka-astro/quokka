if (AMReX_SPACEDIM EQUAL 1)
  add_executable(test_radhydro_pulse_MG_int test_radhydro_pulse_MG_int.cpp ../fextract.cpp ${QuokkaObjSources})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(test_radhydro_pulse_MG_int)
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

	# define variable WORK_PATH = {WORKSPACE}/sims
	set(WORK_PATH "${CMAKE_SOURCE_DIR}/tests/20240408-MG-int-v8.2")

	# mkdir -p WORK_PATH
	file(MAKE_DIRECTORY ${WORK_PATH})

  add_test(NAME RadhydroPulseMGint COMMAND test_radhydro_pulse_MG_int ../RadhydroPulse.in WORKING_DIRECTORY ${WORK_PATH})
endif()
