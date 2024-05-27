if (AMReX_SPACEDIM EQUAL 2)
  add_executable(test_static_sphere_MG2 test_static_sphere_MG2.cpp ../fextract.cpp ${QuokkaObjSources})

  if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(test_static_sphere_MG2)
  endif(AMReX_GPU_BACKEND MATCHES "CUDA")

  add_test(NAME StaticSphereMG2 COMMAND test_static_sphere_MG2 StaticSphereMG.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
