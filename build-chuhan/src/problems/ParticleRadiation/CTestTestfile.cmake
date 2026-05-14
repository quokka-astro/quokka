# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/ParticleRadiation
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/ParticleRadiation
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ParticleRadiation "/Users/meow/quokka/build-chuhan/src/problems/ParticleRadiation/ParticleRadiation" "../inputs/ParticleRadiation.toml")
set_tests_properties(ParticleRadiation PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleRadiation/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/ParticleRadiation/CMakeLists.txt;0;")
add_test(ParticleRadiationLog "/Users/meow/quokka/build-chuhan/src/problems/ParticleRadiation/ParticleRadiation" "../inputs/ParticleRadiationLog.toml")
set_tests_properties(ParticleRadiationLog PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleRadiation/CMakeLists.txt;13;add_test;/Users/meow/quokka/src/problems/ParticleRadiation/CMakeLists.txt;0;")
add_test(ParticleRadiationFastlog "/Users/meow/quokka/build-chuhan/src/problems/ParticleRadiation/ParticleRadiation" "../inputs/ParticleRadiationFastlog.toml")
set_tests_properties(ParticleRadiationFastlog PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleRadiation/CMakeLists.txt;17;add_test;/Users/meow/quokka/src/problems/ParticleRadiation/CMakeLists.txt;0;")
