quokka_add_problem(JOB_NAME HydroContact)

find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME HydroDebugOutputTest
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_hydro_debug.py
            $<TARGET_FILE:HydroContact> ${PROJECT_SOURCE_DIR}/inputs/HydroContact.toml ${AMReX_SPACEDIM})
