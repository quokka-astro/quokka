
add_executable(test_radiation_marshak_cgs test_radiation_marshak_cgs.cpp ../fextract.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_marshak_cgs)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME MarshakWaveCGS COMMAND test_radiation_marshak_cgs MarshakCGS.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
