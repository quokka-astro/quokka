if (AMReX_SPACEDIM EQUAL 1)
  add_executable(test_radhydro_pulse_dyn test_radhydro_pulse_dyn.cpp ../fextract.cpp ${QuokkaObjSources})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(test_radhydro_pulse_dyn)
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME RadhydroPulseDyn COMMAND test_radhydro_pulse_dyn RadhydroPulseDyn.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
