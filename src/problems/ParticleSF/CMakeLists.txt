if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME ParticleSF)
  add_test(NAME ParticleSF2 COMMAND ParticleSF ../inputs/ParticleSF2.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
