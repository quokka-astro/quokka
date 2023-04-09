add_executable(test_hydro_highmach test_hydro_highmach.cpp ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp ../fextract.cpp ../interpolate.cpp ${primordial_chem_sources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_highmach)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroHighMach COMMAND test_hydro_highmach HighMach.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
