add_executable(test_radiation_tube test_radiation_tube.cpp ../fextract.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_tube)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadiationPressureTube COMMAND test_radiation_tube RadTube.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
