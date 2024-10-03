# Include all subdirectories under ../../../extern/Microphysics/integration
include_directories(../../../extern/Microphysics/integration)
include_directories(../../../extern/Microphysics/integration/VODE)
include_directories(../../../extern/Microphysics/integration/utils)

add_executable(test_radiation_SuOlson test_radiation_SuOlson.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_SuOlson)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME SuOlsonTest COMMAND test_radiation_SuOlson SuOlson.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
