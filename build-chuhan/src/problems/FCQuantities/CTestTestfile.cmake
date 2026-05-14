# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/FCQuantities
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/FCQuantities
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(FCQuantities "/Users/meow/quokka/build-chuhan/src/problems/FCQuantities/FCQuantities" "../inputs/FCQuantities.toml")
set_tests_properties(FCQuantities PROPERTIES  COST "0" LABELS "MHD-TEST;MHD-ASAN" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ProblemHelpers.cmake;60;add_test;/Users/meow/quokka/src/problems/FCQuantities/CMakeLists.txt;2;quokka_add_problem;/Users/meow/quokka/src/problems/FCQuantities/CMakeLists.txt;0;")
