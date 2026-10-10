if (AMReX_SPACEDIM EQUAL 3)
    quokka_add_problem(JOB_NAME Turbulence)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/startup)
    add_test(NAME TurbulenceStartup
        COMMAND Turbulence ${QuokkaCode_SOURCE_DIR}/inputs/TurbulenceStartup.toml ${QuokkaTestParams}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/startup)
    add_executable(TurbulenceAmplitudeControllerTest
        ${QuokkaCode_SOURCE_DIR}/tests/test_turbulence_amplitude_controller.cpp)
    add_test(NAME TurbulenceAmplitudeController COMMAND TurbulenceAmplitudeControllerTest)
    add_executable(TurbGenProportionalTest
        ${QuokkaCode_SOURCE_DIR}/tests/test_turbgen_proportional.cpp)
    # Match the production target's SYSTEM treatment of vendored TurbGen headers.
    target_include_directories(TurbGenProportionalTest SYSTEM PRIVATE
        ${QuokkaCode_SOURCE_DIR}/extern/turbulence)
    add_test(NAME TurbGenProportional COMMAND TurbGenProportionalTest)
endif()
