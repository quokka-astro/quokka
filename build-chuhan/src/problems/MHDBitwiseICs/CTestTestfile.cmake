# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/MHDBitwiseICs
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/MHDBitwiseICs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MHDBitwiseICs "/Users/meow/quokka/build-chuhan/src/problems/MHDBitwiseICs/MHDBitwiseICs" "../inputs/MHDBitwiseICs.toml")
set_tests_properties(MHDBitwiseICs PROPERTIES  COST "0" LABELS "MHD-TEST;MHD-ASAN" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ProblemHelpers.cmake;60;add_test;/Users/meow/quokka/src/problems/MHDBitwiseICs/CMakeLists.txt;2;quokka_add_problem;/Users/meow/quokka/src/problems/MHDBitwiseICs/CMakeLists.txt;0;")
