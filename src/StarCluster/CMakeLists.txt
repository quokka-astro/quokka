if (AMReX_SPACEDIM EQUAL 3)
    # Define a custom target that runs the Python script to produce the input perturbations file
    execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/perturbation.py --kmin=2 --kmax=64 --size=128 --alpha=2 --f_solenoidal=1.0
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    )

    execute_process(COMMAND cp -p ${CMAKE_CURRENT_SOURCE_DIR}/zdrv.hdf5 ${CMAKE_BINARY_DIR}/
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    execute_process(COMMAND cp -p ${CMAKE_CURRENT_SOURCE_DIR}/zdrv.hdf5 ${CMAKE_SOURCE_DIR}/tests/
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    message("zdrv.hdf5 created in ${CMAKE_SOURCE_DIR}/tests/ and ${CMAKE_BINARY_DIR}")

    # Create a custom target that depends on the Python script's output
    add_custom_target(RunPythonScript DEPENDS ${CMAKE_SOURCE_DIR}/tests/zdrv.hdf5)

    add_executable(star_cluster star_cluster.cpp TurbDataReader.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(star_cluster)
    endif()

    add_test(NAME StarCluster COMMAND star_cluster StarCluster.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME StarClusterAMR COMMAND star_cluster StarCluster_AMR.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    add_dependencies(star_cluster RunPythonScript)
endif()
