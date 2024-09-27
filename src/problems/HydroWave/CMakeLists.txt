add_executable(test_hydro_wave test_hydro_wave.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_wave)
endif()

add_test(NAME HydroWave COMMAND ${MPI_RUNNER} test_hydro_wave hydro_wave.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
