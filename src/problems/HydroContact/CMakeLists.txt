quokka_add_problem(JOB_NAME HydroContact)
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME CheckpointSentinelTest
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_checkpoint_sentinel.py
    $<TARGET_FILE:HydroContact> ${CMAKE_SOURCE_DIR}/inputs/HydroContact.toml)
