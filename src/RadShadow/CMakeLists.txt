
if (AMReX_SPACEDIM GREATER_EQUAL 2)
    add_executable(test_radiation_shadow ../main.cpp ../CloudyCooling.cpp ../GrackleDataReader.cpp test_radiation_shadow.cpp)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_radiation_shadow)
    endif()
endif()
