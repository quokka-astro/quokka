
add_executable(test_radiation_SuOlson test_radiation_SuOlson.cpp ../fextract.cpp ../interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_SuOlson)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME SuOlsonTest COMMAND test_radiation_SuOlson SuOlson.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
