add_executable(test_advection_se test_advection_semiellipse.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_advection_se)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ScalarAdvectionSemiEllipse COMMAND ${MPI_RUNNER} test_advection_se advection_semiellipse.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
