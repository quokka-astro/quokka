add_executable(test_rad_line_cooling_MG test_rad_line_cooling_MG.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_rad_line_cooling_MG)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadLineCoolingMG COMMAND test_rad_line_cooling_MG ../inputs/RadLineCooling.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME RadLineCoolingMG_coupled COMMAND test_rad_line_cooling_MG ../inputs/RadLineCoolingCoupled.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
