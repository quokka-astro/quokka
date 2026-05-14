# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/RadLineCoolingMG
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/RadLineCoolingMG
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(RadLineCoolingMG "/Users/meow/quokka/build-chuhan/src/problems/RadLineCoolingMG/RadLineCoolingMG" "../inputs/RadLineCoolingMG.toml")
set_tests_properties(RadLineCoolingMG PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/RadLineCoolingMG/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/RadLineCoolingMG/CMakeLists.txt;0;")
add_test(RadLineCoolingMG_coupled "/Users/meow/quokka/build-chuhan/src/problems/RadLineCoolingMG/RadLineCoolingMG" "../inputs/RadLineCoolingCoupled.toml")
set_tests_properties(RadLineCoolingMG_coupled PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/RadLineCoolingMG/CMakeLists.txt;13;add_test;/Users/meow/quokka/src/problems/RadLineCoolingMG/CMakeLists.txt;0;")
