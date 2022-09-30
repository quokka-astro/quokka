add_executable(test_hydro_leblanc ../main.cpp test_hydro_leblanc.cpp ../fextract.cpp ../interpolate.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_leblanc)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroLeblanc COMMAND test_hydro_leblanc leblanc.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
