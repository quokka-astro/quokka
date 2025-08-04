if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_mhd_blast test_mhd_blast.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_mhd_blast)
    endif()

    add_test(NAME MHDBlast COMMAND test_mhd_blast ../inputs/blast_mhd.in ${QuokkaTestParams} ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
