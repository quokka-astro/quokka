if(AMReX_SPACEDIM EQUAL 3)
	setup_target_for_microphysics_compilation(${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/")
	quokka_add_problem(JOB_NAME ParticleDeposition INPUT_FILE ParticleDeposition.toml)

	add_test(NAME ParticleDepositionSliceDiag COMMAND ParticleDeposition ../inputs/ParticleDepositionSlice.toml ${QuokkaTestParams}
		 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

	add_test(NAME ParticleDepositionRejectsUnknownRuntimeOutput
		 COMMAND ${CMAKE_COMMAND}
			 -DQUOKKA_TEST_BINARY=$<TARGET_FILE:ParticleDeposition>
			 -DQUOKKA_TEST_INPUT=../inputs/ParticleDeposition.toml
			 -DQUOKKA_TEST_WORKDIR=${CMAKE_SOURCE_DIR}/tests
			 -DQUOKKA_TEST_OUTPUT_NAME=particle.CIC.typo_density
			 -P ${CMAKE_CURRENT_SOURCE_DIR}/CheckUnknownRuntimeOutput.cmake)
endif()
