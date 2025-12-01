# Add self-gravity test problem (only if gravity is enabled)
if(QUOKKA_GRAVITY)
    if(AMReX_SPACEDIM EQUAL 3)
        setup_target_for_microphysics_compilation(
            ${microphysics_network_name} 
            "${CMAKE_CURRENT_BINARY_DIR}/"
        )

        add_executable(test_self_gravity test_self_gravity.cpp ${QuokkaObjSources} ${gamma_law_sources})
        target_link_libraries(test_self_gravity ${QuokkaObjLinkLibs})

        # C++17 is required due to std::string_view usage
        set_target_properties(test_self_gravity PROPERTIES CXX_STANDARD 17)

        # Copy input file to binary directory
        file(COPY "test_self_gravity.in" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

        add_test(NAME SelfGravityTest COMMAND test_self_gravity test_self_gravity.in WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(SelfGravityTest PROPERTIES TIMEOUT 600 PROCESSORS 4)
    else()
        message(STATUS "SelfGravityTest only works with AMReX_SPACEDIM=3")
    endif()
else()
    message(STATUS "Skipping SelfGravityTest (QUOKKA_GRAVITY=OFF)")
endif()