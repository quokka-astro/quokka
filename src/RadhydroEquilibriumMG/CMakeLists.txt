if (AMReX_SPACEDIM EQUAL 2)
  add_executable(test_radhydro_equilibrium_MG test_radhydro_equilibrium_MG.cpp ../fextract.cpp ${QuokkaObjSources})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(test_radhydro_equilibrium_MG)
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

	# define variable WORK_PATH = {WORKSPACE}/sims
	# set(WORK_PATH "${CMAKE_SOURCE_DIR}/tests/20240408-MG-int-v8.2")

	# mkdir -p WORK_PATH
	# file(MAKE_DIRECTORY ${WORK_PATH})

  add_test(NAME RadhydroEquilibriumMG COMMAND test_radhydro_equilibrium_MG RadhydroEquilibrium.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
