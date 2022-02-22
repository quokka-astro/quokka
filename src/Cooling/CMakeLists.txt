if (AMReX_SPACEDIM GREATER_EQUAL 2)
add_executable(test_cooling ../main.cpp test_cooling.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_cooling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ShockCooling COMMAND test_cooling Cooling.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()