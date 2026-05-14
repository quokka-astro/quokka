# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/HydroWave
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/HydroWave
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(HydroWave "/Users/meow/quokka/build-chuhan/src/problems/HydroWave/HydroWave" "../inputs/HydroWave.toml")
set_tests_properties(HydroWave PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/HydroWave/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/HydroWave/CMakeLists.txt;0;")
add_test(HydroWaveFc "/Users/meow/quokka/build-chuhan/src/problems/HydroWave/HydroWave" "../inputs/HydroWaveFc.toml")
set_tests_properties(HydroWaveFc PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/HydroWave/CMakeLists.txt;13;add_test;/Users/meow/quokka/src/problems/HydroWave/CMakeLists.txt;0;")
