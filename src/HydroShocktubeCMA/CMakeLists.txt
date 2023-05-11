setup_target_for_microphysics_compilation(${microphysics_network_name} ${microphysics_output_dir})

add_executable(test_hydro_shocktube_cma test_hydro_shocktube_cma.cpp ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp ../fextract.cpp ../interpolate.cpp ${gamma_law_sources})

include_directories(${gamma_law_dirs} ${microphysics_output_dir} "includes/extern_parameters.H" "includes/network_properties.H")

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_shocktube_cma)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroShocktubeCMA COMMAND test_hydro_shocktube_cma shocktube_cma.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
