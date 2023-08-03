if (AMReX_SPACEDIM EQUAL 3)
    # Define a custom target that runs the Python script to produce the input perturbations file
    add_custom_command(
        OUTPUT ${CMAKE_SOURCE_DIR}/tests
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/perturbation.py --kmin=2 --kmax=64 --size=128 --alpha=2 --f_solenoidal=1.0
        COMMAND cp -p ${CMAKE_CURRENT_SOURCE_DIR}/zdrv.hdf5 ${CMAKE_SOURCE_DIR}/tests
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/perturbation.py --kmin=2 --kmax=64 --size=128 --alpha=2 --f_solenoidal=1.0
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    )

    add_executable(star_cluster star_cluster.cpp TurbDataReader.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(star_cluster)
    endif()

    add_test(NAME StarCluster COMMAND star_cluster StarCluster.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME StarClusterAMR COMMAND star_cluster StarCluster_AMR.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
