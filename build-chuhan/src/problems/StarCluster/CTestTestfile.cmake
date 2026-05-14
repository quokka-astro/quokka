# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/StarCluster
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/StarCluster
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ComputeStarClusterPerturbations "python3" "/Users/meow/quokka/src/turbulence/perturbation.py" "--kmin=2" "--kmax=64" "--size=128" "--alpha=2" "--f_solenoidal=1.0")
set_tests_properties(ComputeStarClusterPerturbations PROPERTIES  FIXTURES_SETUP "StarCluster_fixture" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/StarCluster/CMakeLists.txt;19;add_test;/Users/meow/quokka/src/problems/StarCluster/CMakeLists.txt;0;")
add_test(StarCluster "/Users/meow/quokka/build-chuhan/src/problems/StarCluster/StarCluster" "../inputs/StarCluster.toml")
set_tests_properties(StarCluster PROPERTIES  FIXTURES_REQUIRED "StarCluster_fixture" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/StarCluster/CMakeLists.txt;25;add_test;/Users/meow/quokka/src/problems/StarCluster/CMakeLists.txt;0;")
