add_executable(test_radhydro_shock_cgs test_radhydro_shock_cgs.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_shock_cgs)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroShockCGS COMMAND test_radhydro_shock_cgs radshock.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
