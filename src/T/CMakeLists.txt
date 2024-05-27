add_executable(T T.cpp ../fextract.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(T)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME T COMMAND T T.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
