add_executable(test_hydro_sms ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_hydro_sms.cpp ../fextract.cpp ${primordial_chem_sources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_sms)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroSlowMovingShock COMMAND test_hydro_sms SlowMovingShock.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
