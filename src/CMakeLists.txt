add_subdirectory(${TwoMomentRad_SOURCE_DIR}/src/kokkos ${TwoMomentRad_BINARY_DIR}/kokkos)
find_package(PythonLibs REQUIRED)

add_executable(test_radiation_matter_coupling radiation_system.cpp hyperbolic_system.cpp test_radiation_matter_coupling.cpp)

add_definitions(-DFMT_HEADER_ONLY)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})

add_executable(test_radiation_marshak radiation_system.cpp hyperbolic_system.cpp test_radiation_marshak.cpp)

add_definitions(-DFMT_HEADER_ONLY)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})

add_executable(test_radiation_classical_marshak radiation_system.cpp hyperbolic_system.cpp test_radiation_classical_marshak.cpp)

add_definitions(-DFMT_HEADER_ONLY)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Kokkos_INCLUDE_DIRS_RET})

target_link_libraries(test_radiation_matter_coupling ${PYTHON_LIBRARY})
target_link_libraries(test_radiation_matter_coupling kokkos)

target_link_libraries(test_radiation_marshak ${PYTHON_LIBRARY})
target_link_libraries(test_radiation_marshak kokkos)

target_link_libraries(test_radiation_classical_marshak ${PYTHON_LIBRARY})
target_link_libraries(test_radiation_classical_marshak kokkos)

include(CTest)
add_test(NAME MatterEnergyExchange COMMAND test_radiation_matter_coupling)
add_test(NAME MarshakWave COMMAND test_radiation_marshak)
add_test(NAME ClassicalMarshakWave COMMAND test_radiation_classical_marshak)
