if (AMReX_SPACEDIM EQUAL 3)
  	set(JOB_NAME BinaryOrbitCIC)

		add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
		if(AMReX_GPU_BACKEND MATCHES "CUDA")
			setup_target_for_cuda_compilation(${JOB_NAME})
		endif()

    add_test(NAME BinaryOrbitCICSplit COMMAND ${JOB_NAME} ../inputs/BinaryOrbit_split.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCICRefactorInit COMMAND ${JOB_NAME} ../inputs/BinaryOrbit_refactor_init.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCICRefactor COMMAND ${JOB_NAME} ../inputs/BinaryOrbit_refactor.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCICRefactorInit2 COMMAND ${JOB_NAME} ../inputs/BinaryOrbit_refactor_init.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCICRefactor2 COMMAND ${JOB_NAME} ../inputs/BinaryOrbit_refactor_splitparticle.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME BinaryOrbitCICAMR COMMAND ${JOB_NAME} ../inputs/BinaryOrbitAMR.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # BinaryOrbitCICRefactor runs after BinaryOrbitCICRefactorInit, even if it fails
    set_tests_properties(BinaryOrbitCICRefactor PROPERTIES DEPENDS "BinaryOrbitCICRefactorInit")
    set_tests_properties(BinaryOrbitCICRefactorInit2 PROPERTIES DEPENDS "BinaryOrbitCICRefactor")
    set_tests_properties(BinaryOrbitCICRefactor2 PROPERTIES DEPENDS "BinaryOrbitCICRefactorInit2")
endif()
