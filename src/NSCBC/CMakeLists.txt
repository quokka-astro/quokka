add_executable(test_channel_flow channel.cpp ${QuokkaObjSources} ../fextract.cpp)
add_executable(test_vortex vortex.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_channel_flow)
    setup_target_for_cuda_compilation(test_vortex)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ChannelFlow COMMAND test_channel_flow NSCBC_Channel.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
