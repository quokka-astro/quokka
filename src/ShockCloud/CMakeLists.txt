if (AMReX_SPACEDIM GREATER_EQUAL 3)
    add_executable(shock_cloud ../main.cpp cloud.cpp ../simulation.cpp ../IntervalsParser.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp)
    add_executable(test_cloudy ../main.cpp test_cloudy.cpp ../simulation.cpp ../IntervalsParser.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(shock_cloud)
        setup_target_for_cuda_compilation(test_cloudy)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 3)