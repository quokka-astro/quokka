add_executable(test_advection test_advection.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_advection)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ScalarAdvection COMMAND ${MPI_RUNNER} test_advection advection_sawtooth.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
