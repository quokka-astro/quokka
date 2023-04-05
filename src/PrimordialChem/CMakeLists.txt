add_executable(test_primordial_chem test_primordial_chem.cpp ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp ${primordial_chem_sources})
target_include_directories(test_primordial_chem PUBLIC ${primordial_chem_dirs} ${CMAKE_BINARY_DIR} "includes/extern_parameters.H")

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_primordial_chem)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME PrimordialChem COMMAND test_primordial_chem unit_test.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
