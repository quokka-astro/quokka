add_executable(test_radiation_force ../main.cpp test_radiation_force.cpp ../fextract.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_force)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")
