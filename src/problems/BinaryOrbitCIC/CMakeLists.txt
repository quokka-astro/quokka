if (AMReX_SPACEDIM EQUAL 3)
    add_executable(binary_orbit binary_orbit.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(binary_orbit)
    endif()

    add_test(NAME BinaryOrbitCICSplit COMMAND binary_orbit BinaryOrbit_split.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCICRefactorInit COMMAND binary_orbit BinaryOrbit_refactor_init.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCICRefactor COMMAND binary_orbit BinaryOrbit_refactor.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCIC COMMAND binary_orbit BinaryOrbit.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCICAMR COMMAND binary_orbit BinaryOrbitAMR.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # BinaryOrbitCICRefactor runs after BinaryOrbitCICRefactorInit, even if it fails
    set_tests_properties(BinaryOrbitCICRefactor PROPERTIES DEPENDS "BinaryOrbitCICRefactorInit")
endif()
