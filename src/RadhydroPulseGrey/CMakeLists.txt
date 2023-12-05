add_executable(test_radhydro_pulse_grey test_radhydro_pulse_grey.cpp ../fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_pulse_grey)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroPulseGrey COMMAND test_radhydro_pulse_grey RadhydroPulseGrey.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
