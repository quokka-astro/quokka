add_executable(test_radiation_matter_coupling_rsla test_radiation_matter_coupling_rsla.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_matter_coupling_rsla)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME MatterEnergyExchangeRSLA COMMAND ${MPI_RUNNER} test_radiation_matter_coupling_rsla MatterEnergyExchangeRSLA.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
