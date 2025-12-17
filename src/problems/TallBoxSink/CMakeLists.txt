if (AMReX_SPACEDIM EQUAL 3)
    add_executable(TallBoxSink testTallBoxSink.cpp ../../turbulence/TurbDataReader.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(TallBoxSink)
    endif()
    add_test(NAME TallBoxSink COMMAND TallBoxSink ../inputs/TallBoxSink.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

		# extract data files for regression testing
		file(
			ARCHIVE_EXTRACT INPUT
			${QuokkaCode_SOURCE_DIR}/extern/problems/TallBoxSink/zdrv.hdf5.zip
			DESTINATION ${QuokkaCode_SOURCE_DIR}/extern/problems/TallBoxSink)
endif()
