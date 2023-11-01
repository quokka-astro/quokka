add_executable(test_radiation_force_multigroup test_radiation_force_multigroup.cpp ../fextract.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_force_multigroup)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadForceMultigroup COMMAND test_radiation_force_multigroup RadForce.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
