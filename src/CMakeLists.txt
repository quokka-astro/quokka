if(QUOKKA_PYTHON) 
  message(STATUS "Building Quokka with Python support (to disable, add
  '-DQUOKKA_PYTHON=OFF' to the CMake command-line options)")
  find_package(PythonLibs)
else()
  message(STATUS "Building Quokka *without* Python support")
endif()

if(PythonLibs_FOUND)
    add_compile_definitions(HAVE_PYTHON)
    include_directories(${PYTHON_INCLUDE_DIRS})
    link_libraries(${PYTHON_LIBRARY})
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
    # only works with gcc!
    include(CodeCoverage)
    # `pip3 install fastcov --user`
    setup_target_for_coverage_fastcov(
        NAME coverage
        EXECUTABLE ${CMAKE_CTEST_COMMAND} -j2
        BASE_DIRECTORY "../"
        EXCLUDE "${PROJECT_SOURCE_DIR}/extern" "${PROJECT_SOURCE_DIR}/src/test_" "include/openmpi/ompi/mpi/cxx/" "/usr/"
        POST_CMD perl -i -pe s!${PROJECT_SOURCE_DIR}/!!g coverage.json && fastcov_to_sonarqube coverage.json
    )
    SET(CMAKE_CXX_FLAGS "-O1")
    SET(CMAKE_C_FLAGS "-O1")
    append_coverage_compiler_flags()
endif()

include_directories(${amrex_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})
include_directories(${CMAKE_SOURCE_DIR}/src)

link_libraries(AMReX::amrex)
link_libraries(fmt::fmt)

include(CTest)

add_subdirectory(Advection)
add_subdirectory(Advection2D)
add_subdirectory(AdvectionSemiellipse)

add_subdirectory(HydroBlast2D)
add_subdirectory(HydroBlast3D)
add_subdirectory(HydroContact)
add_subdirectory(HydroKelvinHelmholz)
add_subdirectory(HydroLeblanc)
add_subdirectory(HydroRichtmeyerMeshkov)
add_subdirectory(HydroShocktube)
add_subdirectory(HydroShuOsher)
add_subdirectory(HydroSMS)
add_subdirectory(HydroVacuum)
add_subdirectory(HydroWave)
add_subdirectory(HydroQuirk)

add_subdirectory(RadBeam)
add_subdirectory(RadForce)
add_subdirectory(RadMarshak)
add_subdirectory(RadMarshakCGS)
add_subdirectory(RadMarshakAsymptotic)
add_subdirectory(RadMatterCoupling)
add_subdirectory(RadMatterCouplingRSLA)
add_subdirectory(RadPulse)
add_subdirectory(RadShadow)
add_subdirectory(RadStreaming)
add_subdirectory(RadSuOlson)
add_subdirectory(RadTophat)
add_subdirectory(RadTube)

add_subdirectory(RadhydroShell)
add_subdirectory(RadhydroShock)
add_subdirectory(RadhydroShockCGS)

add_subdirectory(Cooling)
