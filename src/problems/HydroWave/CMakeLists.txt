add_executable(test_hydro_wave test_hydro_wave.cpp ${QuokkaObjSources})
add_executable(test_hydro_wave_fc test_hydro_wave_fc.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_wave)
    setup_target_for_cuda_compilation(test_hydro_wave_fc)
endif()

add_test(NAME HydroWave COMMAND test_hydro_wave hydro_wave.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroWaveFC COMMAND test_hydro_wave hydro_wave_fc.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)