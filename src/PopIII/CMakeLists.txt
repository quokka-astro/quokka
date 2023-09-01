if (AMReX_SPACEDIM EQUAL 3)
    # Define a custom target that runs the Python script to produce the input perturbations file


    set(microphysics_network_name "primordial_chem") #this will override network_name to primordial_chem for this directory only
    setup_target_for_microphysics_compilation(${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/")

    #use the BEFORE keyword so that these files get priority in compilation for targets in this directory
    #this is critical to ensure the correct Microphysics files are linked to primordial chem target
    include_directories(BEFORE ${primordial_chem_dirs} "${CMAKE_CURRENT_BINARY_DIR}/" "includes/extern_parameters.H" "includes/network_properties.H")

    add_executable(popiii popiii.cpp TurbDataReader.cpp ../main.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp ../Chemistry.cpp ${primordial_chem_sources})
    target_compile_definitions(popiii PUBLIC PRIMORDIAL_CHEM) #this will add #define PRIMORDIAL_CHEM
    
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(popiii)
    endif()

    execute_process(
        COMMAND Python3::Interpreter -c "h5py"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_QUIET
    )

    add_test(NAME ComputePerturbations COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/perturbation.py --kmin=2 --kmax=64 --size=128 --alpha=2 --f_solenoidal=1.0 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME PopIII COMMAND popiii PopIII.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(ComputePerturbations PROPERTIES FIXTURES_SETUP PopIII_fixture)
    set_tests_properties(PopIII PROPERTIES FIXTURES_REQUIRED PopIII_fixture)

    # AMR test only works on Setonix because Gadi and avatar do not have enough memory per GPU
    # add_test(NAME PopIIIAMR COMMAND popiii popiii_AMR.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
