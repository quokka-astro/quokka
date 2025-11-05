set(JOB_NAME HydroWave)

add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif()

add_executable(test_hydro_wave_convergence test_hydro_wave_convergence.cpp
                                           ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(test_hydro_wave_convergence)
endif()

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(
  NAME HydroWaveFC
  COMMAND ${JOB_NAME} ../inputs/hydro_wave_fc.in ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(
  NAME HydroWaveConvergence
  COMMAND test_hydro_wave_convergence ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
