if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME TracerUniformFlow ADD_TEST OFF)
  foreach(order IN ITEMS 1 2)
    foreach(layout IN ITEMS single split amr)
      set(level 0)
      set(grid_size 32)
      if(NOT layout STREQUAL "single")
        set(grid_size 16)
      endif()
      if(layout STREQUAL "amr")
        set(level 1)
      endif()
      add_test(NAME TracerUniformFlow_${order}_${layout}
        COMMAND TracerUniformFlow ${PROJECT_SOURCE_DIR}/inputs/TracerUniformFlow.toml
        hydro.rk_integrator_order=${order} amr.max_grid_size=${grid_size} amr.max_level=${level} ${QuokkaTestParams}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    endforeach()
    add_test(NAME TracerUniformFlow_${order}_physical
      COMMAND TracerUniformFlow ${PROJECT_SOURCE_DIR}/inputs/TracerUniformFlow.toml
      hydro.rk_integrator_order=${order} "quokka.bc=foextrap foextrap foextrap" tracer_test.offset=0.0 ${QuokkaTestParams}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  endforeach()
endif()
