if (AMReX_SPACEDIM EQUAL 3)
    add_executable(jeans_collapse jeans.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(jeans_collapse)
    endif()

    add_test(NAME JeansCollapse COMMAND jeans_collapse Jeans.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
