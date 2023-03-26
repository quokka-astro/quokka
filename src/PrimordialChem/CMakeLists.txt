add_executable(test_primordial_chem test_primordial_chem.cpp ../main.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_primordial_chem)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME PrimordialChem COMMAND test_primordial_chem primordial_chem.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
