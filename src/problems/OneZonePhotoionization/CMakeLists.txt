
set(JOB_NAME OneZonePhotoionization)

# the photochemistry network: src/networks/photoionization_H_caseB_handwritten/network.hpp
include_directories(BEFORE ${CMAKE_SOURCE_DIR}/src/networks/photoionization_H_caseB_handwritten)

add_executable(
  ${JOB_NAME} testOneZonePhotoionization.cpp "${QuokkaSourcesNoEOS}")

# this will add #define PHOTOCHEMISTRY
target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME OneZonePhotoionizationNoEnergy COMMAND ${JOB_NAME} ../inputs/OneZonePhotoionization.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME OneZonePhotoionizationWithEnergy COMMAND ${JOB_NAME} ../inputs/OneZonePhotoionizationWithEnergy.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

