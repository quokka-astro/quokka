add_executable(test_radhydro_uniform_advecting test_radhydro_uniform_advecting.cpp ../fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_uniform_advecting)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroUniformAdvecting COMMAND test_radhydro_uniform_advecting RadhydroUniformAdvecting.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
