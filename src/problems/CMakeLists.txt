# Include helper functions for problems
include(${CMAKE_CURRENT_SOURCE_DIR}/ProblemHelpers.cmake)

# List of directories to skip
set(IGNORE_DIRS
)

# Get all children of the current directory (relative paths)
file(GLOB children CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/*)

# Loop through all children and add them as subdirectories if they contain their own CMakeLists.txt
foreach(child ${children})

  # Skip non-directories
  if(NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
    continue()
  endif()

  # Skip ignored directories
  if(child IN_LIST IGNORE_DIRS)
    continue()
  endif()

  # Add only if directory contains its own CMakeLists.txt
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${child}/CMakeLists.txt)
    add_subdirectory(${child})
  endif()

endforeach()
