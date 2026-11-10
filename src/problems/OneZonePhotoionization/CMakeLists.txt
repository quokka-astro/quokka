set(JOB_NAME OneZonePhotoionization)

quokka_add_problem(JOB_NAME ${JOB_NAME} ADD_TEST OFF)

add_test(NAME OneZonePhotoionizationNoEnergy COMMAND ${JOB_NAME} ../inputs/OneZonePhotoionization.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME OneZonePhotoionizationWithEnergy COMMAND ${JOB_NAME} ../inputs/OneZonePhotoionizationWithEnergy.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
