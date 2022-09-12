# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

#set(CMAKE_C_COMPILER "amdclang" CACHE PATH "")
#set(CMAKE_CXX_COMPILER "amdclang++" CACHE PATH "")
set(CMAKE_C_COMPILER "amdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "hipcc" CACHE PATH "")
set(AMReX_GPU_BACKEND HIP CACHE STRING "")
set(AMReX_AMD_ARCH gfx908 CACHE STRING "") # MI100
#set(ENV{Ascent_DIR} "/g/data/jh2/bw0729/ascent_build/0.9.0-dev-raja+umpire/install/ascent-develop")
option(QUOKKA_PYTHON OFF)
