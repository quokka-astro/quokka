add_executable(test_mhd_prelim ../main.cpp test_mhd_prelim.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_mhd_prelim)
endif()

add_test(NAME MHDPrelim COMMAND test_mhd_prelim hydro_wave.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
