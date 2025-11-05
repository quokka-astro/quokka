if (AMReX_SPACEDIM EQUAL 3)
set(JOB_NAME ParticleAccretion)

    add_executable(${JOB_NAME} test${JOB_NAME}.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(${JOB_NAME})
    endif()

		add_test(NAME ${JOB_NAME} COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
		add_test(NAME ParticleAccretionBH COMMAND ${JOB_NAME} ../inputs/ParticleAccretionBH.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
