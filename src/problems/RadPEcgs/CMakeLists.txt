add_executable(test_pe_cgs test_pe_cgs.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_pe_cgs)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME PEcgs COMMAND test_pe_cgs PEcgs.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
# add_test(NAME PEcgsNodust COMMAND test_pe_cgs PEcgsNodust.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
# add_test(NAME PEcgsGamma1e-5 COMMAND test_pe_cgs PEcgsGamma1e-5.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
# add_test(NAME PEcgsGAmma1e5 COMMAND test_pe_cgs PEcgsGamma1e5.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
