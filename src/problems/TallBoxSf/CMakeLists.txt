if (AMReX_SPACEDIM EQUAL 3)
    add_executable(TallBoxSf testTallBoxSf.cpp ../../turbulence/TurbDataReader.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(TallBoxSf)
    endif()

    # Generate turbulence perturbations before running simulation
    add_test(
      NAME ComputeTallBoxSfPerturbations
      COMMAND python3 ${CMAKE_SOURCE_DIR}/src/turbulence/perturbation.py
              --kmin=1 --kmax=3 --size=128 --alpha=2 --f_solenoidal=1.0
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    add_test(NAME TallBoxSf COMMAND TallBoxSf ../inputs/TallBoxSf.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # Ensure perturbations are computed before simulation runs
    set_tests_properties(ComputeTallBoxSfPerturbations PROPERTIES FIXTURES_SETUP TallBoxSf_fixture)
    set_tests_properties(TallBoxSf PROPERTIES FIXTURES_REQUIRED TallBoxSf_fixture)
    set_tests_properties(TallBoxSf PROPERTIES COST 85)
endif()
