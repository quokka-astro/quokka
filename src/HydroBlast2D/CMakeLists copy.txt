if (AMReX_SPACEDIM EQUAL 2)
  add_executable(test_static_sphere_MG test_static_sphere_MG.cpp ../fextract.cpp ${QuokkaObjSources})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(test_static_sphere_MG)
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME StaticSphereMG COMMAND test_static_sphere_MG StaticSphereMG.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
