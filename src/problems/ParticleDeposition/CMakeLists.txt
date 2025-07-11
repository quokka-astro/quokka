setup_target_for_microphysics_compilation(${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/")

add_executable(test_particle_deposition test_particle_deposition.cpp ${QuokkaObjSources})
target_link_libraries(test_particle_deposition PUBLIC ${QuokkaObjLibs})
target_include_directories(test_particle_deposition PUBLIC ${QuokkaObjIncludes})
target_include_directories(test_particle_deposition PUBLIC ${gamma_law_dirs})
target_include_directories(test_particle_deposition PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

if(QUOKKA_PYTHON)
    target_link_libraries(test_particle_deposition PUBLIC ${Python_LIBRARIES})
endif()

# Test for particle deposition utilities
add_test(NAME ParticleDeposition COMMAND test_particle_deposition tests/test_particle_deposition.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(ParticleDeposition PROPERTIES TIMEOUT 60 PROCESSORS 1)