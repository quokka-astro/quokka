## host configuration
## run with `cmake -C host_config.cmake ..` from inside build directory
##

set(CMAKE_C_COMPILER "gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
option(QUOKKA_PYTHON OFF)
