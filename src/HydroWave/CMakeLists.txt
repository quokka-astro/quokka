add_executable(test_hydro_wave test_hydro_wave.cpp ../fextract.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_wave)
endif()

add_test(NAME HydroWave COMMAND test_hydro_wave hydro_wave.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
