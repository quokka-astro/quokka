# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/ResampledCoolingTest
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/ResampledCoolingTest
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ResampledCoolingTest "/Users/meow/quokka/build-chuhan/src/problems/ResampledCoolingTest/ResampledCoolingTest" "../inputs/ResampledCoolingTest.toml")
set_tests_properties(ResampledCoolingTest PROPERTIES  COST "0" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ProblemHelpers.cmake;60;add_test;/Users/meow/quokka/src/problems/ResampledCoolingTest/CMakeLists.txt;1;quokka_add_problem;/Users/meow/quokka/src/problems/ResampledCoolingTest/CMakeLists.txt;0;")
add_test(ResampledCoolingTestV2 "/Users/meow/quokka/build-chuhan/src/problems/ResampledCoolingTest/ResampledCoolingTest" "../inputs/ResampledCoolingTestV2.toml")
set_tests_properties(ResampledCoolingTestV2 PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ResampledCoolingTest/CMakeLists.txt;3;add_test;/Users/meow/quokka/src/problems/ResampledCoolingTest/CMakeLists.txt;0;")
