if (AMReX_SPACEDIM EQUAL 3)
    add_executable(RadGravPartCluster RadGravPartCluster.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(RadGravPartCluster)
    endif()

    add_test(NAME RadGravPartCluster COMMAND RadGravPartCluster ../inputs/RadGravPartCluster.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
