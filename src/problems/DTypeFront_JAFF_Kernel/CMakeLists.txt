if(AMReX_SPACEDIM EQUAL 3)
  # DTypeFront_JAFF with runtime source options: kernel radius (stromgen.kernel_radius, in cells) and the reduced flux of the
  # injected radiation, radially outward (stromgen.radial_flux, 0 to 1).
  # Shares DTypeFront_JAFF's cache variables for the tableau and c_hat.
  set(JOB_NAME DTypeFront_JAFF_Kernel)
  quokka_add_problem(JOB_NAME ${JOB_NAME} ADD_TEST OFF)
  target_compile_definitions(${JOB_NAME} PRIVATE DTYPEFRONT_JAFF_TABLEAU=${DTypeFront_JAFF_TABLEAU} C_HAT_DIVISOR=${DTypeFront_JAFF_C_HAT_DIVISOR})
  if(DTypeFront_JAFF_NO_RADIATION)
    target_compile_definitions(${JOB_NAME} PRIVATE DTYPEFRONT_JAFF_NO_RADIATION)
  endif()
endif()
