if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_turbulence test_turbulence.cpp ${QuokkaObjSources}
            test_turbulence.cpp
            test_turbulence.cpp)
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_turbulence)
    endif()

    add_test(NAME turbulence COMMAND test_turbulence turbulence.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/inputs)
endif()
