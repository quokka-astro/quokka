# The problem is planar, so it runs unchanged in 3D: the slab is uniform across y and z and every transverse
# column repeats the 1D solution, which is what makes it a useful check that the radiation, dust and
# photochemistry paths behave identically in 3D. 2D is skipped only to keep CI time down.
# One input serves both dimensionalities; only the x entries matter in 1D.
if(AMReX_SPACEDIM EQUAL 1 OR AMReX_SPACEDIM EQUAL 3)
  quokka_add_problem(JOB_NAME DTypeFront1D)
endif()
