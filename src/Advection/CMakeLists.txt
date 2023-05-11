setup_target_for_microphysics_compilation(${microphysics_network_name} ${microphysics_output_dir})
include_directories(${gamma_law_dirs} ${microphysics_output_dir} "includes/extern_parameters.H" "includes/network_properties.H")

add_executable(test_advection test_advection.cpp ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp ../fextract.cpp ${gamma_law_sources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_advection)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ScalarAdvection COMMAND test_advection advection_sawtooth.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
