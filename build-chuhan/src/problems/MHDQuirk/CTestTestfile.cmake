# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/MHDQuirk
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/MHDQuirk
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MHDQuirk "/Users/meow/quokka/build-chuhan/src/problems/MHDQuirk/MHDQuirk" "../inputs/MHDQuirk.toml")
set_tests_properties(MHDQuirk PROPERTIES  COST "0" LABELS "MHD-TEST" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ProblemHelpers.cmake;60;add_test;/Users/meow/quokka/src/problems/MHDQuirk/CMakeLists.txt;2;quokka_add_problem;/Users/meow/quokka/src/problems/MHDQuirk/CMakeLists.txt;0;")
