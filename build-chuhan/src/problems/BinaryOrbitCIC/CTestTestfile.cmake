# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/BinaryOrbitCIC
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/BinaryOrbitCIC
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(BinaryOrbitCICSplit "/Users/meow/quokka/build-chuhan/src/problems/BinaryOrbitCIC/BinaryOrbitCIC" "../inputs/BinaryOrbit_split.toml")
set_tests_properties(BinaryOrbitCICSplit PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;0;")
add_test(BinaryOrbitCICRefactorInit "/Users/meow/quokka/build-chuhan/src/problems/BinaryOrbitCIC/BinaryOrbitCIC" "../inputs/BinaryOrbit_refactor_init.toml")
set_tests_properties(BinaryOrbitCICRefactorInit PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;10;add_test;/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;0;")
add_test(BinaryOrbitCICRefactor "/Users/meow/quokka/build-chuhan/src/problems/BinaryOrbitCIC/BinaryOrbitCIC" "../inputs/BinaryOrbit_refactor.toml")
set_tests_properties(BinaryOrbitCICRefactor PROPERTIES  DEPENDS "BinaryOrbitCICRefactorInit" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;11;add_test;/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;0;")
add_test(BinaryOrbitCICRefactorInit2 "/Users/meow/quokka/build-chuhan/src/problems/BinaryOrbitCIC/BinaryOrbitCIC" "../inputs/BinaryOrbit_refactor_init.toml")
set_tests_properties(BinaryOrbitCICRefactorInit2 PROPERTIES  DEPENDS "BinaryOrbitCICRefactor" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;12;add_test;/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;0;")
add_test(BinaryOrbitCICRefactor2 "/Users/meow/quokka/build-chuhan/src/problems/BinaryOrbitCIC/BinaryOrbitCIC" "../inputs/BinaryOrbit_refactor_splitparticle.toml")
set_tests_properties(BinaryOrbitCICRefactor2 PROPERTIES  DEPENDS "BinaryOrbitCICRefactorInit2" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;13;add_test;/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;0;")
add_test(BinaryOrbitCIC "/Users/meow/quokka/build-chuhan/src/problems/BinaryOrbitCIC/BinaryOrbitCIC" "../inputs/BinaryOrbitCIC.toml")
set_tests_properties(BinaryOrbitCIC PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;14;add_test;/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;0;")
add_test(BinaryOrbitCICAMR "/Users/meow/quokka/build-chuhan/src/problems/BinaryOrbitCIC/BinaryOrbitCIC" "../inputs/BinaryOrbitAMR.toml")
set_tests_properties(BinaryOrbitCICAMR PROPERTIES  WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;15;add_test;/Users/meow/quokka/src/problems/BinaryOrbitCIC/CMakeLists.txt;0;")
