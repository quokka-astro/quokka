if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME MHDAlfvenWaveLinear)

  add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()

  # Default test: non-grid-aligned wavefront (modes 1,1,0), isotropic domain, default EMF scheme
  add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

  # Scheme coverage tests: grid-aligned wavefront (modes 1,0,0), slab domain, explicit EMF schemes
  add_test(NAME ${JOB_NAME}_Q26_B25   COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}_Q26_B25.toml   ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}_FS17_LD04 COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}_FS17_LD04.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${JOB_NAME}_B25_B25   COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}_B25_B25.toml   ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

  set_tests_properties(${JOB_NAME} ${JOB_NAME}_Q26_B25 ${JOB_NAME}_FS17_LD04 ${JOB_NAME}_B25_B25 PROPERTIES LABELS "MHD;MHD-ASAN")
endif()
