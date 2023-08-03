if (AMReX_SPACEDIM EQUAL 3)
    add_executable(star_cluster star_cluster.cpp TurbDataReader.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(star_cluster)
    endif()

    add_test(NAME StarCluster COMMAND star_cluster StarCluster.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME StarClusterAMR COMMAND star_cluster StarCluster_AMR.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
