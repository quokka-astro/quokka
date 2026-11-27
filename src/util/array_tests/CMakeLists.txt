add_executable(ArrayUtilTest testArrayUtil.cpp)
target_include_directories(ArrayUtilTest PRIVATE ..)
target_compile_features(ArrayUtilTest PRIVATE cxx_std_20)
add_test(NAME ArrayUtilTest COMMAND ArrayUtilTest)
