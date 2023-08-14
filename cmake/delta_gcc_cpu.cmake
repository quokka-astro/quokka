# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(AMReX_GPU_BACKEND NONE CACHE STRING "")
set(AMReX_SPACEDIM 3 CACHE STRING "")
set(AMReX_ASCENT OFF CACHE BOOL "")
set(AMReX_CONDUIT OFF CACHE BOOL "")

option(QUOKKA_PYTHON OFF)
set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4 -O0 -ggdb -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-gdwarf-4 -O2 -ggdb -DNDEBUG" CACHE STRING "" FORCE)
