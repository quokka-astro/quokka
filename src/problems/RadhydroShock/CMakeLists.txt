add_executable(test_radhydro_shock test_radhydro_shock.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_shock)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroShock COMMAND test_radhydro_shock radshock_dimensionless.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
