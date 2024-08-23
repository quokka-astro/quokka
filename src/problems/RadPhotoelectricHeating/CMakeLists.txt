add_executable(test_photoelectric_heating test_photoelectric_heating.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_photoelectric_heating)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME PhotoelectricHeating COMMAND test_photoelectric_heating PhotoelectricHeating.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME PhotoelectricHeatingNodust COMMAND test_photoelectric_heating PhotoelectricHeatingNodust.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME PhotoelectricHeatingGamma1e-5 COMMAND test_photoelectric_heating PhotoelectricHeatingGamma1e-5.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME PhotoelectricHeatingGAmma1e5 COMMAND test_photoelectric_heating PhotoelectricHeatingGamma1e5.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
