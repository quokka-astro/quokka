# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/RadMarshakAsymptotic
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/RadMarshakAsymptotic
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(RadMarshakAsymptotic "/Users/meow/quokka/build-chuhan/src/problems/RadMarshakAsymptotic/RadMarshakAsymptotic" "../inputs/RadMarshakAsymptotic.toml")
set_tests_properties(RadMarshakAsymptotic PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/RadMarshakAsymptotic/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/RadMarshakAsymptotic/CMakeLists.txt;0;")
add_test(MarshakWaveAsymptoticCorr "/Users/meow/quokka/build-chuhan/src/problems/RadMarshakAsymptotic/RadMarshakAsymptotic" "../inputs/MarshakAsymptoticCorr.toml")
set_tests_properties(MarshakWaveAsymptoticCorr PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/RadMarshakAsymptotic/CMakeLists.txt;14;add_test;/Users/meow/quokka/src/problems/RadMarshakAsymptotic/CMakeLists.txt;0;")
