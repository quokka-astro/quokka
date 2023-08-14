add_executable(test_hydro_contact test_hydro_contact.cpp ../fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_contact)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroContact COMMAND test_hydro_contact contact_wave.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
