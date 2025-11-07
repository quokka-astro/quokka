add_executable(test_hydro_wave test_hydro_wave.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_wave)
endif()

add_executable(test_hydro_wave_convergence test_hydro_wave_convergence.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_wave_convergence)
endif()

add_test(NAME HydroWave COMMAND test_hydro_wave ../inputs/hydro_wave.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroWaveFC COMMAND test_hydro_wave ../inputs/hydro_wave_fc.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroWaveConvergence COMMAND test_hydro_wave_convergence ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)