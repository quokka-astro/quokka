add_executable(test_photoelectric_heating test_photoelectric_heating.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_photoelectric_heating)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME PhotoelectricHeating COMMAND test_photoelectric_heating PhotoelectricHeating.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
