add_executable(test_hydro_wave_fc test_hydro_wave_fc.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_wave_fc)
endif()

add_test(NAME HydroWaveFC COMMAND test_hydro_wave_fc hydro_wave_fc.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
