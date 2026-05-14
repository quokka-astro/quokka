# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/ParticleCreation
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/ParticleCreation
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ParticleCreation "/Users/meow/quokka/build-chuhan/src/problems/ParticleCreation/ParticleCreation" "../inputs/ParticleCreation.toml")
set_tests_properties(ParticleCreation PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleCreation/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/ParticleCreation/CMakeLists.txt;0;")
add_test(ParticleCreationAMR "/Users/meow/quokka/build-chuhan/src/problems/ParticleCreation/ParticleCreation" "../inputs/ParticleCreationAMR.toml")
set_tests_properties(ParticleCreationAMR PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleCreation/CMakeLists.txt;13;add_test;/Users/meow/quokka/src/problems/ParticleCreation/CMakeLists.txt;0;")
