if (AMReX_SPACEDIM EQUAL 3)
    add_executable(sink_spherical_collapse sink_spherical_collapse.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(sink_spherical_collapse)
    endif()

    add_test(NAME SinkSphericalCollapse COMMAND sink_spherical_collapse SinkSphericalCollapse.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
