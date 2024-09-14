# Include all subdirectories under ../../../extern/Microphysics/integration
include_directories(../../../extern/Microphysics/integration)
include_directories(../../../extern/Microphysics/integration/VODE)
include_directories(../../../extern/Microphysics/integration/utils)

add_executable(test_radhydro_uniform_advecting test_radhydro_uniform_advecting.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_uniform_advecting)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroUniformAdvecting COMMAND test_radhydro_uniform_advecting RadhydroUniformAdvecting.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
