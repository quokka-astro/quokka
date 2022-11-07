add_executable(test_advection_se test_advection_semiellipse.cpp ../main.cpp ../MFInterpolater.cpp ../fextract.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_advection_se)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ScalarAdvectionSemiEllipse COMMAND test_advection_se advection_semiellipse.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)