if (AMReX_SPACEDIM EQUAL 3)
    add_executable(mass_conserv mass_conserv.cpp ../../util/fextract.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(mass_conserv)
    endif()

    add_test(NAME MassConserv COMMAND mass_conserv MassConserv.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
