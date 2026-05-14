# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/TallBoxSf
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/TallBoxSf
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ComputeTallBoxSfPerturbations "python3" "/Users/meow/quokka/src/turbulence/perturbation.py" "--kmin=1" "--kmax=3" "--size=128" "--alpha=2" "--f_solenoidal=1.0")
set_tests_properties(ComputeTallBoxSfPerturbations PROPERTIES  FIXTURES_SETUP "TallBoxSf_fixture" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/TallBoxSf/CMakeLists.txt;8;add_test;/Users/meow/quokka/src/problems/TallBoxSf/CMakeLists.txt;0;")
add_test(TallBoxSf "/Users/meow/quokka/build-chuhan/src/problems/TallBoxSf/TallBoxSf" "../inputs/TallBoxSf.toml")
set_tests_properties(TallBoxSf PROPERTIES  COST "85" FIXTURES_REQUIRED "TallBoxSf_fixture" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/TallBoxSf/CMakeLists.txt;14;add_test;/Users/meow/quokka/src/problems/TallBoxSf/CMakeLists.txt;0;")
