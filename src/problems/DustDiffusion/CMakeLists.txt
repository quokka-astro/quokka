add_executable(test_dust_diffusion test_dust_diffusion.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_dust_diffusion)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME DustDiffusion COMMAND test_dust_diffusion ../inputs/DustDiffusion.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
