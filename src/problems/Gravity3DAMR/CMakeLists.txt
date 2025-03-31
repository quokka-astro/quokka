if (AMReX_SPACEDIM EQUAL 3)
    add_executable(gravity_3d_amr gravity_3d_amr.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(gravity_3d_amr)
    endif()

    add_test(NAME Gravity3DAMR COMMAND gravity_3d_amr Gravity3DAMR.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
