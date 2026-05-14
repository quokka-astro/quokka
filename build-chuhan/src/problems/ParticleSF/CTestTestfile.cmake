# CMake generated Testfile for 
# Source directory: /Users/meow/quokka/src/problems/ParticleSF
# Build directory: /Users/meow/quokka/build-chuhan/src/problems/ParticleSF
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ParticleSF "/Users/meow/quokka/build-chuhan/src/problems/ParticleSF/ParticleSF" "../inputs/ParticleSF.toml")
set_tests_properties(ParticleSF PROPERTIES  COST "100" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleSF/CMakeLists.txt;9;add_test;/Users/meow/quokka/src/problems/ParticleSF/CMakeLists.txt;0;")
add_test(ParticleSF2 "/Users/meow/quokka/build-chuhan/src/problems/ParticleSF/ParticleSF" "../inputs/ParticleSF2.toml")
set_tests_properties(ParticleSF2 PROPERTIES  COST "99" DEPENDS "ParticleSF" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleSF/CMakeLists.txt;10;add_test;/Users/meow/quokka/src/problems/ParticleSF/CMakeLists.txt;0;")
add_test(ParticleSF_restart_cap "/Users/meow/quokka/build-chuhan/src/problems/ParticleSF/ParticleSF" "../inputs/ParticleSF_restart_cap.toml")
set_tests_properties(ParticleSF_restart_cap PROPERTIES  COST "99" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleSF/CMakeLists.txt;11;add_test;/Users/meow/quokka/src/problems/ParticleSF/CMakeLists.txt;0;")
add_test(ParticleSF_restart_cap2 "/Users/meow/quokka/build-chuhan/src/problems/ParticleSF/ParticleSF" "../inputs/ParticleSF_restart_cap2.toml")
set_tests_properties(ParticleSF_restart_cap2 PROPERTIES  COST "98" DEPENDS "ParticleSF_restart_cap" WORKING_DIRECTORY "/Users/meow/quokka/tests" _BACKTRACE_TRIPLES "/Users/meow/quokka/src/problems/ParticleSF/CMakeLists.txt;12;add_test;/Users/meow/quokka/src/problems/ParticleSF/CMakeLists.txt;0;")
