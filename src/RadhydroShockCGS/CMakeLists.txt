add_executable(test_radhydro_shock_cgs ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radhydro_shock_cgs.cpp ../fextract.cpp ../interpolate.cpp $<TARGET_OBJECTS:primordialchem>)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_shock_cgs)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroShockCGS COMMAND test_radhydro_shock_cgs radshock.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
