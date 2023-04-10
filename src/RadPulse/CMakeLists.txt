add_executable(test_radiation_pulse ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radiation_pulse.cpp ../fextract.cpp $<TARGET_OBJECTS:primordialchem>)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_pulse)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadiationPulse COMMAND test_radiation_pulse RadPulse.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
