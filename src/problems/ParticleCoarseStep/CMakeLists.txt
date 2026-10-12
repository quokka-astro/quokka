if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME ParticleCoarseStep ADD_TEST OFF)
  foreach(subcycle IN ITEMS 0 1)
    add_test(NAME ParticleCoarseStep_${subcycle}
      COMMAND ParticleCoarseStep ${PROJECT_SOURCE_DIR}/inputs/ParticleCoarseStep.toml
      particle_test.file=${PROJECT_SOURCE_DIR}/inputs/ParticleCoarseStep.txt do_subcycle=${subcycle} ${QuokkaTestParams}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  endforeach()
endif()
