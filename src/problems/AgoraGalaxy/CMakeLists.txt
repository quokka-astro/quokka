if (AMReX_SPACEDIM EQUAL 3)
    add_executable(agora_galaxy galaxy.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(agora_galaxy)
    endif()

    add_test(NAME AgoraGalaxy COMMAND agora_galaxy AgoraGalaxy.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
