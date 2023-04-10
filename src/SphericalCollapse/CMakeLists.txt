if (AMReX_SPACEDIM EQUAL 3)
    add_executable(spherical_collapse ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp spherical_collapse.cpp)
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(spherical_collapse)
    endif()

    add_test(NAME SphericalCollapse COMMAND spherical_collapse SphericalCollapse.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
