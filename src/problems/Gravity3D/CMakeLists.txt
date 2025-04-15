if (AMReX_SPACEDIM EQUAL 3)
    add_executable(gravity_3d gravity_3d.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(gravity_3d)
    endif()

    add_test(NAME Gravity3D COMMAND gravity_3d Gravity3D.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME Gravity3DSplit COMMAND gravity_3d Gravity3D_split.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
