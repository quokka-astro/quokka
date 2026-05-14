# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/PopIII
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/PopIII
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ComputePerturbations "python3" "/Users/meow/quokka/src/turbulence/perturbation.py" "--kmin=2" "--kmax=32" "--size=64" "--alpha=1.8" "--f_solenoidal=0.66667")
set_tests_properties(ComputePerturbations PROPERTIES  FIXTURES_SETUP "PopIII_fixture" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/PopIII/CMakeLists.txt;93;add_test;/Users/meow/quokka/src/problems/PopIII/CMakeLists.txt;0;")
add_test(PopIII "/Users/meow/quokka/build-chuhan/src/problems/PopIII/PopIII" "../inputs/PopIII.toml")
set_tests_properties(PopIII PROPERTIES  FIXTURES_REQUIRED "PopIII_fixture" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/PopIII/CMakeLists.txt;100;add_test;/Users/meow/quokka/src/problems/PopIII/CMakeLists.txt;0;")
