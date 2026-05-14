# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/RadMarshakDustPE
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/RadMarshakDustPE
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(RadMarshakDustPE "/Users/meow/quokka/build-chuhan/src/problems/RadMarshakDustPE/RadMarshakDustPE" "../inputs/RadMarshakDustPE.toml")
set_tests_properties(RadMarshakDustPE PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/RadMarshakDustPE/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/RadMarshakDustPE/CMakeLists.txt;0;")
add_test(RadiationMarshakDustPE-decoupled "/Users/meow/quokka/build-chuhan/src/problems/RadMarshakDustPE/RadMarshakDustPE" "../inputs/RadMarshakDustPEdecoupled.toml")
set_tests_properties(RadiationMarshakDustPE-decoupled PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/RadMarshakDustPE/CMakeLists.txt;13;add_test;/Users/meow/quokka/src/problems/RadMarshakDustPE/CMakeLists.txt;0;")
