add_executable(test_hydro_momentum_sponge test_hydro_momentum_sponge.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_momentum_sponge)
endif()

add_test(NAME HydroMomentumSponge COMMAND test_hydro_momentum_sponge ../inputs/momentum_sponge.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

