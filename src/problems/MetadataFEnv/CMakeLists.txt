quokka_add_problem(JOB_NAME MetadataFEnv ADD_TEST OFF)
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME MetadataFEnvTest COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_metadata_fenv.py
  $<TARGET_FILE:MetadataFEnv> ${PROJECT_SOURCE_DIR}/inputs/HydroContact.toml)
