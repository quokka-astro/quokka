
add_executable(test_linear_diffusion_multigroup test_linear_diffusion_multigroup.cpp ../fextract.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_linear_diffusion_multigroup)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadLinearDiffusionMultigroup COMMAND test_linear_diffusion_multigroup LinearDiffusion.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
