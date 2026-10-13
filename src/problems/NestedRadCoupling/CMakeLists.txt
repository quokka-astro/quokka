quokka_add_problem(JOB_NAME NestedRadCoupling)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(NestedRadCoupling PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
endif()
