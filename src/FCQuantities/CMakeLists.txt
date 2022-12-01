add_executable(test_fc_quantities ../main.cpp test_fc_quantities.cpp ../interpolate.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_fc_quantities)
endif()

add_test(NAME FCQuantities COMMAND test_fc_quantities fc_hydro_wave.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
