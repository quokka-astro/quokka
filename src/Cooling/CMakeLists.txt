
find_package(SUNDIALS REQUIRED)

add_executable(test_cooling ../main.cpp test_cooling.cpp ../fextract.cpp)
target_link_libraries(test_cooling PUBLIC SUNDIALS::cvode)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_cooling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME SundialsCooling COMMAND test_cooling Cooling.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
