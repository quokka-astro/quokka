if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_brio_wu_shocktube test_brio_wu_shocktube.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_brio_wu_shocktube)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")

    add_test(NAME BrioWuShockTube COMMAND test_brio_wu_shocktube ../inputs/shocktube_mhd.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(BrioWuShockTube PROPERTIES LABELS "MHD")
endif()