if (AMReX_SPACEDIM EQUAL 3)
    add_executable(binary_orbit binary_orbit.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(binary_orbit)
    endif()

    add_test(NAME BinaryOrbitCIC COMMAND binary_orbit BinaryOrbit.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
