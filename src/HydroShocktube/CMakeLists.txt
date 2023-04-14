add_executable(test_hydro_shocktube ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_hydro_shocktube.cpp ../fextract.cpp ../interpolate.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_shocktube)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroShocktube COMMAND test_hydro_shocktube shocktube.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
