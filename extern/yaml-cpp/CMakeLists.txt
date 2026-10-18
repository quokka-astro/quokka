# Minimal build of the vendored, parse-only subset of yaml-cpp (see QUOKKA_VENDOR.md).
file(GLOB yaml-cpp-sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(yaml-cpp STATIC ${yaml-cpp-sources})
add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)

target_include_directories(yaml-cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(yaml-cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(yaml-cpp PUBLIC YAML_CPP_STATIC_DEFINE PRIVATE YAML_CPP_NO_CONTRIB)
set_target_properties(yaml-cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
