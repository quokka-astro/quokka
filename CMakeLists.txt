cmake_minimum_required(VERSION 3.16)
project(QuokkaCode VERSION 1.0
  DESCRIPTION "Radiation hydrodynamics with structured AMR"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
#set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")
#set(CMAKE_C_COMPILER_LAUNCHER "ccache")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/extern/amrex/Tools/CMake)
include(CTest)

set(AMReX_SPACEDIM 1 CACHE STRING "")
set(AMReX_MPI ON CACHE BOOL "" FORCE)
set(AMReX_HDF5 OFF CACHE BOOL "" FORCE) # do not use
set(AMReX_LINEAR_SOLVERS ON CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS_INCFLO ON CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS_EM OFF CACHE BOOL "" FORCE)
set(AMReX_CONDUIT OFF CACHE BOOL "") # optional
set(AMReX_ASCENT OFF CACHE BOOL "") # optional
set(AMReX_AMRLEVEL OFF CACHE BOOL "" FORCE)
set(AMReX_PROBINIT OFF CACHE BOOL "" FORCE)
set(AMReX_TINY_PROFILE ON CACHE BOOL "" FORCE)

# Configure AMReX-Hydro to match the AMReX build that Quokka uses
if(NOT DEFINED AMReX_GPU_BACKEND)
  set(AMReX_GPU_BACKEND NONE CACHE STRING "" FORCE)
endif()

set(QUOKKA_ENABLE_AMREX_HYDRO OFF)
if(AMReX_SPACEDIM EQUAL 3)
  set(QUOKKA_ENABLE_AMREX_HYDRO ON)
  set(HYDRO_SPACEDIM ${AMReX_SPACEDIM} CACHE STRING "" FORCE)
  set(HYDRO_MPI ${AMReX_MPI} CACHE BOOL "" FORCE)
  set(HYDRO_PROJECTIONS ON CACHE BOOL "" FORCE)
  set(HYDRO_EB OFF CACHE BOOL "" FORCE)
  set(HYDRO_NO_EB ON CACHE BOOL "" FORCE)
  set(HYDRO_GPU_BACKEND ${AMReX_GPU_BACKEND} CACHE STRING "" FORCE)
endif()

## a Python installation can be specified with Python_ROOT_DIR
##   see: https://cmake.org/cmake/help/latest/module/FindPython.html#hints
## NumPy and Matplotlib can be installed with `python3 -m pip install numpy matplotlib --user`
option(QUOKKA_PYTHON "Compile with Python support (on/off)" ON)
option(DISABLE_FMAD "Disable fused multiply-add instructions on GPU (on/off)" ON)
option(ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENABLE_HWASAN "Enable HWAddressSanitizer" OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(QUOKKA_OPENPMD "Enable OpenPMD output (on/off)" OFF)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  enable_language(CUDA)
  
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR "You must use CUDA version 12.0 or newer to compile Quokka because C++20 support is needed.")
  endif()
  
  if(CMAKE_VERSION VERSION_LESS 3.20)
    include(AMReX_SetupCUDA)
  endif(CMAKE_VERSION VERSION_LESS 3.20)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

if(AMReX_GPU_BACKEND MATCHES "HIP")
  execute_process(
      COMMAND hipcc --version
      RESULT_VARIABLE hipcc_result
      OUTPUT_VARIABLE hipcc_version
      ERROR_VARIABLE hipcc_error
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT hipcc_result EQUAL 0)
      message(FATAL_ERROR "Failed to detect HIP compiler version: ${hipcc_error}")
  endif()

  string(REPLACE "HIP version: " "" hipcc_version ${hipcc_version})
  message(STATUS "HIP compiler version: ${hipcc_version}")

  if(hipcc_version VERSION_LESS 6.3)
    message(FATAL_ERROR "You must use ROCm version >= 6.3 to compile Quokka. All previous ROCm versions have compiler bugs that may cause Quokka to produce incorrect results.")
  endif()
endif(AMReX_GPU_BACKEND MATCHES "HIP")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # abort! Intel Compiler Classic cannot compile Quokka!
  message(FATAL_ERROR "You have configured CMake to use the 'classic' Intel compilers (icc/icpc), which are the old Intel compilers. They cannot compile Quokka correctly! You must use the new LLVM-based Intel compilers (icx/icpx) instead by adding the following CMake command-line options: -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

# this is necessary to prevent GCC from warning about ARM64 ABI changes in GCC 10.1
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wno-psabi)
endif()

# this is necessary to avoid spurious FPEs due to vectorization
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
	option(ENABLE_TESTS_FPE "Enable floating-point exceptions when running tests" OFF)
else()
	option(ENABLE_TESTS_FPE "Enable floating-point exceptions when running tests" ON)
endif()

execute_process(COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${QuokkaCode_SOURCE_DIR}
    OUTPUT_VARIABLE QUOKKA_GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${QuokkaCode_SOURCE_DIR}
    OUTPUT_VARIABLE QUOKKA_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${QuokkaCode_SOURCE_DIR}/extern/amrex
    OUTPUT_VARIABLE AMREX_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if the working directory is dirty
execute_process(
    COMMAND git status --porcelain -uno
    WORKING_DIRECTORY ${QuokkaCode_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_STATUS_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set a flag if the working directory is dirty
if(NOT GIT_STATUS_OUTPUT STREQUAL "")
    set(GIT_DIRTY 1)
    message(WARNING "This Quokka repository has untracked changes not reflected in its git history! That means that the exact provenance of plotfiles and checkpoints CANNOT be inferred from their metadata.\nIt is STRONGLY recommended that you commit your changes to a branch locally, then re-run CMake and recompile Quokka in order to obtain reproducible simulation outputs.")
    # Append "-dirty" to the commit hash
    string(APPEND QUOKKA_GIT_HASH "-dirty")
else()
    set(GIT_DIRTY 0)
endif()

# add preprocessor macros for git commits
add_definitions(-DAMREX_GIT_HASH="${AMREX_GIT_HASH}")
add_definitions(-DQUOKKA_GIT_HASH="${QUOKKA_GIT_HASH}")
message(STATUS "Quokka git branch: ${QUOKKA_GIT_BRANCH}")
message(STATUS "Quokka git commit hash: ${QUOKKA_GIT_HASH}")
message(STATUS "AMReX git commit hash: ${AMREX_GIT_HASH}")

add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/amrex ${QuokkaCode_BINARY_DIR}/amrex)
if(AMReX_SPACEDIM EQUAL 3)
  set(AMRVOLUMERENDERER_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
  set(AMRVOLUMERENDERER_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/amrVolumeRenderer ${QuokkaCode_BINARY_DIR}/amrVolumeRenderer)
endif()
if(QUOKKA_ENABLE_AMREX_HYDRO)
  add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/AMReX-Hydro ${QuokkaCode_BINARY_DIR}/AMReX-Hydro)
endif()
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/fmt ${QuokkaCode_BINARY_DIR}/fmt)
if(AMReX_GPU_BACKEND MATCHES "HIP")
  target_compile_definitions(fmt PRIVATE NDEBUG)
endif()
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/yaml-cpp ${QuokkaCode_BINARY_DIR}/yaml-cpp)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/turbulence/plugins/AMReX ${QuokkaCode_BINARY_DIR}/turbulence)

#add compiler definitions again because
#they do not carry forward from Microphysics
#We use Strang operator-spliting while integrating the network
#NAUX_NET defines the number of auxiliary variables
add_definitions(-DNAUX_NET -DSTRANG)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/Microphysics ${QuokkaCode_BINARY_DIR}/Microphysics)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/src ${QuokkaCode_BINARY_DIR}/src)
