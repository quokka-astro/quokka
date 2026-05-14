# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/RadDust
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/RadDust
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(RadDust "mpirun" "-np" "2" "/Users/meow/quokka/build-chuhan/src/problems/RadDust/RadDust" "../inputs/RadDust.toml")
set_tests_properties(RadDust PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/RadDust/CMakeLists.txt;8;add_test;/Users/meow/quokka/src/problems/RadDust/CMakeLists.txt;0;")
