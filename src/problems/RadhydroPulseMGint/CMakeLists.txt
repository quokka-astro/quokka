set(JOB_NAME RadhydroPulseMGint)

add_executable(${JOB_NAME} testRadhydroPulseMGint.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
		setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

# define variable WORK_PATH = {WORKSPACE}/sims
# set(WORK_PATH "${CMAKE_SOURCE_DIR}/tests/20240408-MG-int-v8.2")

# mkdir -p WORK_PATH
# file(MAKE_DIRECTORY ${WORK_PATH})

add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/RadhydroPulse.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
