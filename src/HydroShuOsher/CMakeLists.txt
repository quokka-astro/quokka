add_executable(test_hydro_shuosher ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_hydro_shuosher.cpp ../fextract.cpp ../interpolate.cpp $<TARGET_OBJECTS:primordialchem>)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_shuosher)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ShuOsher COMMAND test_hydro_shuosher ShuOsher.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
