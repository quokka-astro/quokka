add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/amrex ${QuokkaCode_BINARY_DIR}/amrex)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/fmt ${QuokkaCode_BINARY_DIR}/fmt)

find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${amrex_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

link_libraries(${PYTHON_LIBRARY})
link_libraries(AMReX::amrex)
link_libraries(fmt::fmt)

function(add_clang_static_analysis target)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        get_target_property(SRCs ${target} SOURCES)
        add_library(${target}_analyze OBJECT ${SRCs})
        target_include_directories(${target}_analyze PRIVATE ${PYTHON_INCLUDE_DIRS})
        target_include_directories(${target}_analyze PRIVATE ${Kokkos_INCLUDE_DIRS_RET} ${fmt_INCLUDE_DIRS_RET})
        target_compile_options(${target}_analyze BEFORE PRIVATE "--analyze")
    endif()
endfunction()

add_executable(test_radiation_matter_coupling radiation_system.cpp hyperbolic_system.cpp test_radiation_matter_coupling.cpp interpolate.c)
add_executable(test_radiation_matter_coupling_rsla radiation_system.cpp hyperbolic_system.cpp test_radiation_matter_coupling_rsla.cpp interpolate.c)
add_executable(test_radiation_SuOlson radiation_system.cpp hyperbolic_system.cpp test_radiation_SuOlson.cpp interpolate.c)
add_executable(test_radiation_marshak radiation_system.cpp hyperbolic_system.cpp test_radiation_marshak.cpp interpolate.c)
add_executable(test_radiation_marshak_cgs radiation_system.cpp hyperbolic_system.cpp test_radiation_marshak_cgs.cpp interpolate.c)
add_executable(test_radiation_marshak_asymptotic radiation_system.cpp hyperbolic_system.cpp test_radiation_marshak_asymptotic.cpp interpolate.c)
add_executable(test_radhydro_shock radiation_system.cpp hydro_system.cpp hyperbolic_system.cpp test_radhydro_shock.cpp interpolate.c)
add_executable(test_radhydro_shock_cgs radiation_system.cpp hydro_system.cpp hyperbolic_system.cpp test_radhydro_shock_cgs.cpp interpolate.c)
add_executable(test_hydro_shocktube hydro_system.cpp hyperbolic_system.cpp test_hydro_shocktube.cpp interpolate.c)
add_executable(test_hydro_leblanc hydro_system.cpp hyperbolic_system.cpp test_hydro_leblanc.cpp interpolate.c)
add_executable(test_hydro_shuosher hydro_system.cpp hyperbolic_system.cpp test_hydro_shuosher.cpp interpolate.c)
add_executable(test_hydro_sms hydro_system.cpp hyperbolic_system.cpp test_hydro_sms.cpp)
add_executable(test_hydro_contact hydro_system.cpp hyperbolic_system.cpp test_hydro_contact.cpp)
add_executable(test_hydro_vacuum hydro_system.cpp hyperbolic_system.cpp test_hydro_vacuum.cpp interpolate.c)
add_executable(test_advection linear_advection.cpp hyperbolic_system.cpp simulation.cpp test_advection.cpp)
add_executable(test_advection_se linear_advection.cpp hyperbolic_system.cpp test_advection_semiellipse.cpp)
add_executable(test_hydro_wave hydro_system.cpp hyperbolic_system.cpp test_hydro_wave.cpp)
add_executable(test_radiation_pulse hyperbolic_system.cpp radiation_system.cpp test_radiation_pulse.cpp)
add_executable(test_radiation_streaming hyperbolic_system.cpp radiation_system.cpp test_radiation_streaming.cpp)
add_executable(test_amrex_2d test_amrex_2d.cpp diffusion.cpp)

add_clang_static_analysis(test_radiation_matter_coupling)
add_clang_static_analysis(test_radiation_SuOlson)
add_clang_static_analysis(test_radiation_marshak)
add_clang_static_analysis(test_radiation_marshak_cgs)
add_clang_static_analysis(test_radhydro_shock)
add_clang_static_analysis(test_radhydro_shock_cgs)
add_clang_static_analysis(test_hydro_shocktube)
add_clang_static_analysis(test_hydro_leblanc)
add_clang_static_analysis(test_hydro_shuosher)
add_clang_static_analysis(test_advection)
add_clang_static_analysis(test_advection_se)
add_clang_static_analysis(test_hydro_wave)
add_clang_static_analysis(test_radiation_pulse)
add_clang_static_analysis(test_radiation_streaming)

include(CTest)
message(DEBUG "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
add_test(NAME MatterEnergyExchange COMMAND test_radiation_matter_coupling WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME MatterEnergyExchangeRSLA COMMAND test_radiation_matter_coupling_rsla WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME SuOlsonTest COMMAND test_radiation_SuOlson WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME MarshakWave COMMAND test_radiation_marshak WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME MarshakWaveCGS COMMAND test_radiation_marshak_cgs WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME RadhydroShock COMMAND test_radhydro_shock WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME RadhydroShockCGS COMMAND test_radhydro_shock_cgs WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroShocktube COMMAND test_hydro_shocktube WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME ScalarAdvection COMMAND test_advection WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME ScalarAdvectionSemiEllipse COMMAND test_advection_se WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroLeblanc COMMAND test_hydro_leblanc WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroSlowMovingShock COMMAND test_hydro_sms WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroContact COMMAND test_hydro_contact WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroVacuum COMMAND test_hydro_vacuum WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME HydroWave COMMAND test_hydro_wave WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME RadiationStreaming COMMAND test_radiation_streaming WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME RadiationPulse COMMAND test_radiation_pulse WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME ShuOsher COMMAND test_hydro_shuosher WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# include(CodeCoverage)
# append_coverage_compiler_flags()
# setup_target_for_coverage_gcovr_html(
#         NAME coverage                 
#         EXECUTABLE ctest -j ${n_cores}
#         DEPENDENCIES
#             test_radiation_matter_coupling
#             test_radiation_matter_coupling_rsla
#             test_radiation_SuOlson
#             test_radiation_marshak
#             test_radiation_marshak_cgs
#             test_radiation_streaming
#             test_radiation_pulse
#             test_advection
#             test_advection_se
#             test_hydro_shocktube
#             test_hydro_leblanc
#             test_hydro_sms
#             test_hydro_contact
#             test_hydro_vacuum
#             test_hydro_wave
#             test_hydro_shuosher)