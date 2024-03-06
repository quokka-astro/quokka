if (AMReX_SPACEDIM EQUAL 3)
    # Define a custom target that runs the Python script to produce the input perturbations file

    add_executable(star_cluster star_cluster.cpp ../TurbDataReader.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(star_cluster)
    endif()

    execute_process(
        COMMAND Python3::Interpreter -c "h5py"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_QUIET
    )

    add_test(NAME ComputeStarClusterPerturbations COMMAND python3 ${CMAKE_SOURCE_DIR}/src/perturbation.py --kmin=2 --kmax=64 --size=128 --alpha=2 --f_solenoidal=1.0 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME StarCluster COMMAND star_cluster StarCluster.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(ComputeStarClusterPerturbations PROPERTIES FIXTURES_SETUP    StarCluster_fixture)
    set_tests_properties(StarCluster PROPERTIES FIXTURES_REQUIRED StarCluster_fixture)

    # AMR test only works on Setonix because Gadi and avatar do not have enough memory per GPU
    # add_test(NAME StarClusterAMR COMMAND star_cluster StarCluster_AMR.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
