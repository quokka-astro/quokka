add_executable(test_box ../main.cpp test_box.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp ../fextract.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_box)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

if (AMReX_SPACEDIM GREATER_EQUAL 2)
    add_executable(test_cooling ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_cooling.cpp)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_cooling)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 2)