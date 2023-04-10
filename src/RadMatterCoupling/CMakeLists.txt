
add_executable(test_radiation_matter_coupling ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radiation_matter_coupling.cpp ../fextract.cpp ../interpolate.cpp $<TARGET_OBJECTS:primordialchem>)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_matter_coupling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME MatterEnergyExchange COMMAND test_radiation_matter_coupling energyexchange.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
