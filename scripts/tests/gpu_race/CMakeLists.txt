find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME GPURaceScriptTest COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_race.py ${CMAKE_CURRENT_SOURCE_DIR}/../../bash/test_gpu_race.sh)
