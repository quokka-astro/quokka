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

    add_test(
      NAME PrepareTallBoxSfChemOutput
      COMMAND ${CMAKE_COMMAND}
              -DOUTDIR=${CMAKE_SOURCE_DIR}/tests/tallbox_chem
              -P ${CMAKE_SOURCE_DIR}/scripts/cmake/prepare_tallbox_chem_output.cmake)

    add_test(
      NAME PrepareTallBoxSfChemYields
      COMMAND ${CMAKE_COMMAND}
              -DQUOKKA_YIELD_ARCHIVE=${CMAKE_SOURCE_DIR}/extern/yields/yields.tar.gz
              -DQUOKKA_YIELD_DESTINATION=${CMAKE_SOURCE_DIR}/extern/yields
              -DQUOKKA_YIELD_SENTINEL=${CMAKE_SOURCE_DIR}/extern/yields/SNII_Kobayashi0611
              -P ${CMAKE_SOURCE_DIR}/scripts/cmake/extract_chemical_yields.cmake)

    add_test(
      NAME TallBoxSfChem
      COMMAND TallBoxSf ../inputs/TallBoxSf_chem_test.toml ${QuokkaTestParams}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # Ensure perturbations are computed before simulation runs
    set_tests_properties(ComputeTallBoxSfPerturbations PROPERTIES FIXTURES_SETUP TallBoxSf_fixture)
    set_tests_properties(TallBoxSf PROPERTIES FIXTURES_REQUIRED TallBoxSf_fixture)
    set_tests_properties(PrepareTallBoxSfChemOutput PROPERTIES FIXTURES_SETUP TallBoxSfChemOutput_fixture)
    set_tests_properties(PrepareTallBoxSfChemYields PROPERTIES FIXTURES_SETUP TallBoxSfChemYields_fixture)
    set_tests_properties(TallBoxSfChem PROPERTIES FIXTURES_REQUIRED "TallBoxSf_fixture;TallBoxSfChemOutput_fixture;TallBoxSfChemYields_fixture" FIXTURES_SETUP TallBoxSfChem_fixture)
    set_tests_properties(TallBoxSf PROPERTIES COST 85)
    set_tests_properties(TallBoxSfChem PROPERTIES COST 1200 TIMEOUT 7200)
endif()
