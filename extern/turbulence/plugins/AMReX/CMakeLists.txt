cmake_minimum_required(VERSION 3.16)
project(TurbGenEx LANGUAGES CXX)

add_library(turbulence INTERFACE)
add_library(turbulence::turbulence ALIAS turbulence)

target_sources(turbulence INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/TurbGenEx.h
)

target_link_libraries(turbulence INTERFACE amrex)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TURB_INCLUDE_TYPE INTERFACE)
else()
    set(TURB_INCLUDE_TYPE SYSTEM INTERFACE)
endif()

target_include_directories(turbulence ${TURB_INCLUDE_TYPE}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
