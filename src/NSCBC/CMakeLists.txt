add_executable(test_channel_flow channel.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_channel_flow)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")
