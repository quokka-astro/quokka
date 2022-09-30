add_executable(test_radiation_matter_coupling_rsla ../main.cpp test_radiation_matter_coupling_rsla.cpp ../fextract.cpp ../interpolate.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_matter_coupling_rsla)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME MatterEnergyExchangeRSLA COMMAND test_radiation_matter_coupling_rsla MatterEnergyExchangeRSLA.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
