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

# HDF5
find_package(HDF5 REQUIRED)

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
    add_compile_options("-O1")
    append_coverage_compiler_flags()
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  # enable AddressSanitizer when debugging
  add_compile_options(-O1 -fno-omit-frame-pointer)
  add_compile_options(-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment)
  add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

# this is necessary to prevent the host compiler from complaining about nv_pragmas
<<<<<<< HEAD
# only works for gcc
=======
>>>>>>> 939463a... disable -Wno-unknown-pragmas
#add_compile_options(-Wno-unknown-pragmas)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)


include_directories(${amrex_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})
include_directories(${HDF5_INCLUDE_DIRS_RET})
include_directories(${Boost_INCLUDE_DIRS_RET})
include_directories(${CMAKE_SOURCE_DIR}/src)

link_libraries(AMReX::amrex)
link_libraries(fmt::fmt)
link_libraries(hdf5::hdf5)

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

add_subdirectory(ODEIntegration)
add_subdirectory(Cooling)
add_subdirectory(ShockCloud)
