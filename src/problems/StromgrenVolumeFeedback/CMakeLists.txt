if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME StromgrenVolumeFeedback PRIORITY 90)

  # Variants of the same binary, each exercising a different path through the module.
  #   Gradient: non-uniform medium with an off-centre source, where the analytic radius no longer
  #             applies and photon conservation is the only exact statement.
  #   Subgrid:  R_St far below one cell -- the unresolved regime the module exists for.
  #   Overlap:  two co-located sources, checking that photons are never spent twice.
  foreach(_variant Gradient Subgrid Overlap)
    add_test(
      NAME StromgrenVolume${_variant}
      COMMAND StromgrenVolumeFeedback ../inputs/StromgrenVolume${_variant}.toml ${QuokkaTestParams}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(StromgrenVolume${_variant} PROPERTIES COST 90)
  endforeach()
endif()
