add_executable(test_radhydro_shock_multigroup test_radhydro_shock_multigroup.cpp ../fextract.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_shock_multigroup)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroShockCGS COMMAND test_radhydro_shock_multigroup radshockMG.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
