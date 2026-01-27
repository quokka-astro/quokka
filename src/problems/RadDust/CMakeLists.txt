set(JOB_NAME RadDust)

add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
if(AMReX_GPU_BACKEND MATCHES "CUDA")
	setup_target_for_cuda_compilation(${JOB_NAME})
endif()

add_test(NAME ${JOB_NAME} COMMAND mpirun -np 2 $<TARGET_FILE:${JOB_NAME}> ../inputs/${JOB_NAME}.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
