if (AMReX_SPACEDIM EQUAL 1)
  add_executable(test_radhydro_pulse test_radhydro_pulse.cpp ../fextract.cpp ${QuokkaObjSources})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(test_radhydro_pulse)
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME RadhydroPulse COMMAND test_radhydro_pulse RadhydroPulse.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
