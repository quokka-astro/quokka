add_subdirectory(${TwoMomentRad_SOURCE_DIR}/extern/kokkos ${TwoMomentRad_BINARY_DIR}/kokkos)
add_subdirectory(${TwoMomentRad_SOURCE_DIR}/extern/fmt ${TwoMomentRad_BINARY_DIR}/fmt)

find_package(PythonLibs REQUIRED)

add_executable(test_radiation_matter_coupling radiation_system.cpp hyperbolic_system.cpp test_radiation_matter_coupling.cpp interpolate.c)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

add_executable(test_radiation_marshak radiation_system.cpp hyperbolic_system.cpp test_radiation_marshak.cpp interpolate.c)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

add_executable(test_radiation_classical_marshak radiation_system.cpp hyperbolic_system.cpp test_radiation_classical_marshak.cpp interpolate.c)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

add_executable(test_radhydro_shock radiation_system.cpp hydro_system.cpp hyperbolic_system.cpp test_radhydro_shock.cpp interpolate.c)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

add_executable(test_hydro_shocktube hydro_system.cpp hyperbolic_system.cpp test_hydro_shocktube.cpp interpolate.c)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

add_executable(test_advection linear_advection.cpp hyperbolic_system.cpp test_advection.cpp interpolate.c)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

add_executable(test_hydro_wave hydro_system.cpp hyperbolic_system.cpp test_hydro_wave.cpp)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

add_executable(test_radiation_pulse hyperbolic_system.cpp radiation_system.cpp test_radiation_pulse.cpp interpolate.c)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

target_link_libraries(test_radiation_matter_coupling ${PYTHON_LIBRARY})
target_link_libraries(test_radiation_matter_coupling kokkos)
target_link_libraries(test_radiation_matter_coupling fmt::fmt)

target_link_libraries(test_radiation_marshak ${PYTHON_LIBRARY})
target_link_libraries(test_radiation_marshak kokkos)
target_link_libraries(test_radiation_marshak fmt::fmt)

target_link_libraries(test_radiation_classical_marshak ${PYTHON_LIBRARY})
target_link_libraries(test_radiation_classical_marshak kokkos)
target_link_libraries(test_radiation_classical_marshak fmt::fmt)

target_link_libraries(test_radhydro_shock ${PYTHON_LIBRARY})
target_link_libraries(test_radhydro_shock kokkos)
target_link_libraries(test_radhydro_shock fmt::fmt)

target_link_libraries(test_hydro_shocktube ${PYTHON_LIBRARY})
target_link_libraries(test_hydro_shocktube kokkos)
target_link_libraries(test_hydro_shocktube fmt::fmt)

target_link_libraries(test_advection ${PYTHON_LIBRARY})
target_link_libraries(test_advection kokkos)
target_link_libraries(test_advection fmt::fmt)

target_link_libraries(test_hydro_wave ${PYTHON_LIBRARY})
target_link_libraries(test_hydro_wave kokkos)
target_link_libraries(test_hydro_wave fmt::fmt)

target_link_libraries(test_radiation_pulse ${PYTHON_LIBRARY})
target_link_libraries(test_radiation_pulse kokkos)
target_link_libraries(test_radiation_pulse fmt::fmt)

include(CTest)
add_test(NAME MatterEnergyExchange COMMAND test_radiation_matter_coupling)
add_test(NAME MarshakWave COMMAND test_radiation_marshak)
add_test(NAME ClassicalMarshakWave COMMAND test_radiation_classical_marshak)
add_test(NAME RadhydroShock COMMAND test_radhydro_shock)
add_test(NAME HydroShocktube COMMAND test_hydro_shocktube)
add_test(NAME ScalarAdvection COMMAND test_advection)
add_test(NAME HydroWave COMMAND test_hydro_wave)
add_test(NAME RadiationPulse COMMAND test_radiation_pulse)