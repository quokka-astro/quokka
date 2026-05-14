#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "AMReX-Hydro::amrex_hydro_api" for configuration "Release"
set_property(TARGET AMReX-Hydro::amrex_hydro_api APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AMReX-Hydro::amrex_hydro_api PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libamrex_hydro_api.a"
  )

list(APPEND _cmake_import_check_targets AMReX-Hydro::amrex_hydro_api )
list(APPEND _cmake_import_check_files_for_AMReX-Hydro::amrex_hydro_api "${_IMPORT_PREFIX}/lib/libamrex_hydro_api.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
