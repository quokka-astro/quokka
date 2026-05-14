# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/ParticleAccretion
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/ParticleAccretion
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ParticleAccretion "/Users/meow/quokka/build-chuhan/src/problems/ParticleAccretion/ParticleAccretion" "../inputs/ParticleAccretion.toml")
set_tests_properties(ParticleAccretion PROPERTIES  COST "95" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleAccretion/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/ParticleAccretion/CMakeLists.txt;0;")
add_test(ParticleAccretionBH "/Users/meow/quokka/build-chuhan/src/problems/ParticleAccretion/ParticleAccretion" "../inputs/ParticleAccretionBH.toml")
set_tests_properties(ParticleAccretionBH PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleAccretion/CMakeLists.txt;13;add_test;/Users/meow/quokka/src/problems/ParticleAccretion/CMakeLists.txt;0;")
