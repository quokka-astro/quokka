if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME ParticleSplit ADD_TEST OFF)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  add_test(NAME ParticleSplitTest COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_particle_split.py
    $<TARGET_FILE:ParticleSplit> ${PROJECT_SOURCE_DIR}/inputs/ParticleSplit.toml)
endif()
