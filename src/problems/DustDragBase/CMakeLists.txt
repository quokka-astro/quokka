add_executable(test_dust_drag_base test_dust_drag_base.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_dust_drag_base)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME DustDragBase COMMAND test_dust_drag_base ../inputs/DustDragBase.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
