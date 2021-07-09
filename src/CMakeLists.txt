include_directories(${amrex_INCLUDE_DIRS_RET})
link_libraries(AMReX::amrex)

add_executable(test_advection2d main.cpp test_advection2d.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_advection2d)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

include(CTest)
message(DEBUG "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
add_test(NAME ScalarAdvection2D COMMAND test_advection2d WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)