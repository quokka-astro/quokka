setup_target_for_microphysics_compilation(${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/")

quokka_add_problem(JOB_NAME ParticleDeposition INPUT_FILE ParticleDeposition.in)
set_tests_properties(ParticleDeposition PROPERTIES TIMEOUT 60 PROCESSORS 1)
