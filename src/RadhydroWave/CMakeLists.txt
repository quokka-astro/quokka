add_executable(test_radhydro_wave ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radhydro_wave.cpp ../fextract.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_wave)
endif()

add_test(NAME RadhydroWave COMMAND test_radhydro_wave radhydro_wave.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
