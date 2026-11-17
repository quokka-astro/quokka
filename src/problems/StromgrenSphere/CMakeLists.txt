if(AMReX_SPACEDIM EQUAL 3)
  # two executables built from StromgrenSphere.hpp, each with its own test and input
  quokka_add_problem(JOB_NAME StromgrenSphere)
  quokka_add_problem(JOB_NAME StromgrenSphereTempIndependentRecombination)
endif()
