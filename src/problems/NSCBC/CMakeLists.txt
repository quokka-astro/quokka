set(JOB_NAME NSCBC)

add_executable(${JOB_NAME} testNSCBC.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

if (AMReX_SPACEDIM GREATER_EQUAL 2)
    add_executable(test_vortex vortex.cpp ${QuokkaObjSources})
    
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_vortex)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
endif(AMReX_SPACEDIM GREATER_EQUAL 2)
