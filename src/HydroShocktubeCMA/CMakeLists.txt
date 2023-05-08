add_executable(test_hydro_shocktube_cma test_hydro_shocktube_cma.cpp ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp ../fextract.cpp ../interpolate.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_shocktube_cma)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroShocktubeCMA COMMAND test_hydro_shocktube_cma shocktube_cma.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
