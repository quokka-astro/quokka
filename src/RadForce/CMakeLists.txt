add_executable(test_radiation_force ../main.cpp ../MFInterpolater.cpp test_radiation_force.cpp ../fextract.cpp ../interpolate.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_force)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadForce COMMAND test_radiation_force RadForce.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
