
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was AMReX-HydroConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(HYDRO_SPACEDIM 3)
set(HYDRO_EB OFF)
set(HYDRO_MPI ON)
set(HYDRO_OMP OFF)
set(HYDRO_GPU_BACKEND NONE)

find_package(AMReX QUIET REQUIRED )

include("${CMAKE_CURRENT_LIST_DIR}/AMReX-HydroTargets.cmake")

set(AMReX-Hydro_INCLUDE_DIRS "${PROJECT_PREFIX_DIR}/include")
set(AMReX-Hydro_LIBRARY_DIRS "${PROJECT_PREFIX_DIR}/lib")
set(AMReX-Hydro_LIBRARIES "AMReX-Hydro::amrex_hydro_api")

set(AMReX-Hydro_FOUND TRUE)
check_required_components(AMReX-Hydro)
