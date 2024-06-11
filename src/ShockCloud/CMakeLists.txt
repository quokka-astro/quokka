add_executable(test_grackle_cooling test_grackle_cooling.cpp ${QuokkaObjSources})
add_executable(test_cloudy_cooling test_cloudy_cooling.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_grackle_cooling)
    setup_target_for_cuda_compilation(test_cloudy_cooling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME GrackleLikeCooling COMMAND test_grackle_cooling Cooling.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME CloudyCooling COMMAND test_cloudy_cooling TabulatedCooling.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

if (AMReX_SPACEDIM GREATER_EQUAL 3)
    add_executable(shock_cloud cloud.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(shock_cloud)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 3)
