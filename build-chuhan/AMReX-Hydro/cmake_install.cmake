# Install script for directory: /Users/meow/quokka/extern/AMReX-Hydro

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/meow/quokka/build-chuhan/AMReX-Hydro/Utils/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/meow/quokka/build-chuhan/AMReX-Hydro/MOL/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/meow/quokka/build-chuhan/AMReX-Hydro/Godunov/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/meow/quokka/build-chuhan/AMReX-Hydro/BDS/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/meow/quokka/build-chuhan/AMReX-Hydro/Projections/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/meow/quokka/build-chuhan/AMReX-Hydro/libamrex_hydro_api.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libamrex_hydro_api.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libamrex_hydro_api.a")
    execute_process(COMMAND "/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libamrex_hydro_api.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/Users/meow/quokka/extern/AMReX-Hydro/Utils/hydro_utils.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/Utils/hydro_constants.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/Utils/hydro_bcs_K.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/MOL/hydro_mol.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/MOL/hydro_mol_edge_state_K.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/Godunov/hydro_godunov.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/Godunov/hydro_godunov_plm.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/Godunov/hydro_godunov_ppm.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/Godunov/hydro_godunov_corner_couple.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/BDS/hydro_bds.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/Projections/hydro_MacProjector.H"
    "/Users/meow/quokka/extern/AMReX-Hydro/Projections/hydro_NodalProjector.H"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX-Hydro/AMReX-HydroTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX-Hydro/AMReX-HydroTargets.cmake"
         "/Users/meow/quokka/build-chuhan/AMReX-Hydro/CMakeFiles/Export/ea8537aad6186cc1e3e3083b8afd6051/AMReX-HydroTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX-Hydro/AMReX-HydroTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX-Hydro/AMReX-HydroTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX-Hydro" TYPE FILE FILES "/Users/meow/quokka/build-chuhan/AMReX-Hydro/CMakeFiles/Export/ea8537aad6186cc1e3e3083b8afd6051/AMReX-HydroTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX-Hydro" TYPE FILE FILES "/Users/meow/quokka/build-chuhan/AMReX-Hydro/CMakeFiles/Export/ea8537aad6186cc1e3e3083b8afd6051/AMReX-HydroTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX-Hydro" TYPE FILE FILES "/Users/meow/quokka/build-chuhan/AMReX-Hydro/AMReX-HydroConfig.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/meow/quokka/build-chuhan/AMReX-Hydro/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
