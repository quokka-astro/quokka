add_executable(test_radhydro_shock ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radhydro_shock.cpp ../fextract.cpp ../interpolate.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_shock)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroShock COMMAND test_radhydro_shock radshock_dimensionless.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
