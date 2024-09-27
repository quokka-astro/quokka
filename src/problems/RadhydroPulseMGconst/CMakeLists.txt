add_executable(test_radhydro_pulse_MG_const_kappa test_radhydro_pulse_MG_const_kappa.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
		setup_target_for_cuda_compilation(test_radhydro_pulse_MG_const_kappa)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroPulseMGconst COMMAND ${MPI_RUNNER} test_radhydro_pulse_MG_const_kappa RadhydroPulse.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
