if (AMReX_SPACEDIM GREATER_EQUAL 3)
    add_executable(shock_cloud ../main.cpp cloud.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp)
    add_executable(test_cloudy ../main.cpp test_cloudy.cpp ../GrackleDataReader.cpp ../CloudyCooling.cpp)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(shock_cloud)
        target_compile_options(shock_cloud PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>) 

        setup_target_for_cuda_compilation(test_cloudy)
        target_compile_options(test_cloudy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>) 
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 3)