add_executable(test_radiation_marshak_asymptotic ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radiation_marshak_asymptotic.cpp ../fextract.cpp ../interpolate.cpp $<TARGET_OBJECTS:primordialchem>)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_marshak_asymptotic)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

#add_test(NAME MarshakWaveAsymptoticDiffusion COMMAND test_radiation_marshak_asymptotic MarshakAsymptotic.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
