if (AMReX_SPACEDIM EQUAL 2)
	add_executable(test_radiation_streaming_y test_radiation_streaming_y.cpp ../fextract.cpp ${QuokkaObjSources})

	if(AMReX_GPU_BACKEND MATCHES "CUDA")
			setup_target_for_cuda_compilation(test_radiation_streaming_y)
	endif(AMReX_GPU_BACKEND MATCHES "CUDA")

	add_test(NAME RadiationStreamingY COMMAND test_radiation_streaming_y RadStreamingY.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
