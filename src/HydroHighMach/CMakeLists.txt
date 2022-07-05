add_executable(test_hydro_highmach test_hydro_highmach.cpp ../main.cpp ../fextract.cpp ../interpolate.c)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_highmach)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

# disable temporarily for CI
#add_test(NAME HydroHighMach COMMAND test_hydro_highmach HighMach.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
