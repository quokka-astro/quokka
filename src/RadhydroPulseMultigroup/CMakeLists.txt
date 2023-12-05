add_executable(test_radhydro_pulse_multigroup test_radhydro_pulse_multigroup.cpp ../fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_pulse_multigroup)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroPulseMultigroup COMMAND test_radhydro_pulse_multigroup RadhydroPulse.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
