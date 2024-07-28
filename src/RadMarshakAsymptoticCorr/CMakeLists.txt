add_executable(test_radiation_marshak_asymptotic_corr test_radiation_marshak_asymptotic_corr.cpp ../fextract.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_marshak_asymptotic_corr)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME MarshakWaveAsymptoticDiffusionCorr COMMAND test_radiation_marshak_asymptotic_corr MarshakAsymptotic.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
