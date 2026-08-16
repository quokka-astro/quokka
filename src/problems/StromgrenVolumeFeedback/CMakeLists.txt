if(AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME StromgrenVolumeFeedback PRIORITY 90)

  # Variant of the same binary: non-uniform medium with an off-centre source, which checks photon
  # conservation on the path where the analytic Strömgren radius no longer applies.
  add_test(
    NAME StromgrenVolumeGradient
    COMMAND StromgrenVolumeFeedback ../inputs/StromgrenVolumeGradient.toml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  set_tests_properties(StromgrenVolumeGradient PROPERTIES COST 90)
endif()
