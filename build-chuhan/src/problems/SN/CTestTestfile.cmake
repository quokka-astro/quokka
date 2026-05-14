# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/SN
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/SN
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(SN "/Users/meow/quokka/build-chuhan/src/problems/SN/SN" "../inputs/SN.toml")
set_tests_properties(SN PROPERTIES  COST "90" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/SN/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/SN/CMakeLists.txt;0;")
add_test(SN-n1.0 "/Users/meow/quokka/build-chuhan/src/problems/SN/SN" "../inputs/SN.toml" "problem.n_amb=1.0")
set_tests_properties(SN-n1.0 PROPERTIES  COST "89" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/SN/CMakeLists.txt;15;add_test;/Users/meow/quokka/src/problems/SN/CMakeLists.txt;0;")
add_test(SN-n0.1 "/Users/meow/quokka/build-chuhan/src/problems/SN/SN" "../inputs/SN.toml" "problem.n_amb=0.1")
set_tests_properties(SN-n0.1 PROPERTIES  COST "88" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/SN/CMakeLists.txt;21;add_test;/Users/meow/quokka/src/problems/SN/CMakeLists.txt;0;")
add_test(SN-n10-no-smoothing "/Users/meow/quokka/build-chuhan/src/problems/SN/SN" "../inputs/SN.toml" "particles.SN_smooth_gas_velocity=false")
set_tests_properties(SN-n10-no-smoothing PROPERTIES  COST "87" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/SN/CMakeLists.txt;27;add_test;/Users/meow/quokka/src/problems/SN/CMakeLists.txt;0;")
add_test(SN-n1.0-no-smoothing "/Users/meow/quokka/build-chuhan/src/problems/SN/SN" "../inputs/SN.toml" "problem.n_amb=1.0" "particles.SN_smooth_gas_velocity=false")
set_tests_properties(SN-n1.0-no-smoothing PROPERTIES  COST "86" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/SN/CMakeLists.txt;33;add_test;/Users/meow/quokka/src/problems/SN/CMakeLists.txt;0;")
add_test(SN-n0.1-no-smoothing "/Users/meow/quokka/build-chuhan/src/problems/SN/SN" "../inputs/SN.toml" "problem.n_amb=0.1" "particles.SN_smooth_gas_velocity=false")
set_tests_properties(SN-n0.1-no-smoothing PROPERTIES  COST "85" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/SN/CMakeLists.txt;39;add_test;/Users/meow/quokka/src/problems/SN/CMakeLists.txt;0;")
