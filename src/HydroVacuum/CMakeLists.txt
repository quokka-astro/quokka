add_executable(test_hydro_vacuum ../main.cpp test_hydro_vacuum.cpp ../fextract.cpp ../interpolate.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_vacuum)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroVacuum COMMAND test_hydro_vacuum vacuum.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
