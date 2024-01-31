if (AMReX_SPACEDIM EQUAL 1)
  add_executable(test_radhydro_pulse_MG_int test_radhydro_pulse_MG_int.cpp ../fextract.cpp ${QuokkaObjSources})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(test_radhydro_pulse_MG_int)
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME RadhydroPulseMGint COMMAND test_radhydro_pulse_MG_int RadhydroPulse.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
