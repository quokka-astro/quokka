if (AMReX_SPACEDIM EQUAL 3)
	add_executable(test_radparticle_3D test_radparticle_3D.cpp ${QuokkaObjSources})

	if(AMReX_GPU_BACKEND MATCHES "CUDA")
			setup_target_for_cuda_compilation(test_radparticle_3D)
	endif(AMReX_GPU_BACKEND MATCHES "CUDA")

	add_test(NAME RadParticle3D COMMAND test_radparticle_3D ../inputs/RadParticles3D.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
