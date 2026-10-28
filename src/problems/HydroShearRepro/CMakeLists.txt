if(AMReX_SPACEDIM EQUAL 2)
  quokka_add_problem(JOB_NAME HydroShearRepro ADD_TEST OFF)
  # Protect both face admissibility and completion of the formerly crashing run.
  add_test(NAME HydroShearRepro_positive_reconstruction
    COMMAND HydroShearRepro ${CMAKE_SOURCE_DIR}/inputs/HydroShearReproFailure.toml
      ${QuokkaTestParams} shear.check_reconstruction=1)
  add_test(NAME HydroShearRepro_xppm_av0
    COMMAND HydroShearRepro ${CMAKE_SOURCE_DIR}/inputs/HydroShearReproFailure.toml
      ${QuokkaTestParams} hydro.reconstruction_order=5 hydro.artificial_viscosity_coefficient=0
      hydro.use_dual_energy=1 stop_time=3 amr.v=0
      shear.history_file=shear_repro_xppm_av0_history.txt
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  foreach(reconstruction IN ITEMS 3 5)
    add_test(NAME HydroShearRepro_order${reconstruction}
      COMMAND HydroShearRepro ${CMAKE_SOURCE_DIR}/inputs/HydroShearReproFailure.toml
        ${QuokkaTestParams} hydro.reconstruction_order=${reconstruction}
        hydro.artificial_viscosity_coefficient=0.1 hydro.use_dual_energy=1
        stop_time=0.3 amr.v=0
        shear.history_file=shear_repro_order${reconstruction}_history.txt
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  endforeach()
endif()
