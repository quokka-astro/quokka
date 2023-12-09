.. Debugging simulation instability

Debugging simulation instability
=====

Nonlinear stability of systems of PDEs is an unsolved problem. There is no complete, rigorous mathematical theory.
There are two concepts, however, that are closely associated with nonlinear stability:

* *positivity preservation:* This is the property that, given a positive initial density and pressure at timestep :math:`n`,
  the density and pressure are positive at timestep :math:`n+1`. For theoretical background, see
  `Linde & Roe (1997) <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/77032/AIAA-1997-2098-398.pdf?sequence=1>`_.
  and `Perthame & Shu (1996) <https://link.springer.com/article/10.1007/s002110050187>`_.
* *entropy stability:* This is the property that the *discretized* system of equations obeys the second law of thermodynamics,
  i.e. the discrete entropy of the simulation must be non-decreasing. There is also a stronger local form,
  where the entropy variable everywhere obeys an entropy inequality. For theoretical background, see
  `Harten (1983) <https://www.sciencedirect.com/science/article/pii/0021999183901183>`_ and
  `Tadmor (1986) <https://doi.org/10.1016/0168-9274(86)90029-2>`_. (This assumes a convex equation of state.)

If a simulation goes unstable, it is likely due to one of the above properties being violated.
It is important to note that standard finite volume reconstruction methods do *not* guarantee entropy stability
(see `Fjordholm et al. 2012 <https://epubs.siam.org/doi/10.1137/110836961>`_).

It is also possible that the entropy is nondecreasing, but insufficient entropy is produced for a given shock
compared to the amount that should be produced physically. This will cause an unphysical oscillatory solution.

Ways to improve stability
-----------------------
The solution is either to reduce the timestep or add additional dissipation:

* set the initial timestep to be 0.1 or 0.01 of the CFL timestep by setting ``sim.initDt_`` appropriately

* lower the CFL number
  
  * It should be in the range 0.1-0.3. If it's above 0.3, it's linearly unstable, so it will never work.
  * If it's below 0.1, it's sufficiently low that the simulation will be very inefficient.
    If it still doesn't work, experience indicates that reducing it further usually does not help.

* reduce the order of the spatial reconstruction
 
  * By default PPM reconstruction is used, but PLM (with minmod limiter) can be used instead. It is much more dissipative, and therefore, stable.

* re-try the hydro update with a smaller timestep
 
  * This is necessary because the positivity-preserving timestep may be much smaller than the
    CFL-limited timestep near the boundary of realizable states (`Linde & Roe 1997 <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/77032/AIAA-1997-2098-398.pdf?sequence=1>`_).
  * Quokka will do this automatically, but only up to a maximum hard-coded number of retries.
  * If the simulation still fails, this usually indicates a stability problem that will probably not be fixed by further timestep reductions.

* revert to a first-order update in problem cells
 
  * For a sufficiently small timestep, this is provably entropy stable and positivity-preserving (as long as the Riemann solver itself is, which requires robust wavespeeds)
  * Quokka reduces to first-order in space and time automatically when the density is negative in a given cell.
  * In the future, Quokka could be extended to also revert to first-order based on entropy.

* use wavespeed estimates that are robust for strong shocks

  * The eigenvalues of the Roe-average state do not provide correct bounds for very strong shocks.
  * If the shocks at the interface travel faster than the wavespeed estimates, there will be insufficient entropy production.
  * Doing this requires additional assumptions about the EOS
    (`Miller & Puckett 1996 <https://www.sciencedirect.com/science/article/pii/S0021999196902004>`_).
  * Quokka attempts to do this for ideal gases and as well as materials that can be approximated with a
    Mie-Gruniesen EOS (see `Dukowicz 1985 <https://ui.adsabs.harvard.edu/abs/1985JCoPh..61..119D/abstract>`_ and
    `Rider 1999 <https://www.osti.gov/biblio/760447>`_).
  * No code changes should be required unless you are simulating an exotic material or a condensed matter phase transition
    (gaseous phase transitions do not cause any issues; see `Bethe 1942 <https://link.springer.com/chapter/10.1007/978-1-4612-2218-7_11>`_).

* add artificial viscosity

  * This can be helpful because it adds dissipation when shocks are propagating transverse to the interface.
  * For sufficient entropy production, it is important that the velocity divergence estimator
    is based on the cell-average velocities surrounding the interface, *not* the reconstructed velocities.
  * This can be enabled in Quokka with the runtime parameter ``hydro.artificial_viscosity_coefficient``. A value of ``0.1`` is recommended.
  * This parameter is identical to the artificial viscosity coefficient described in
    `Colella and Woodward 1984 <https://ui.adsabs.harvard.edu/abs/1984JCoPh..54..174C/abstract>`_.

Floors
-----------------------
As an absolute last resort, one can enable density and/or temperature floors for a simulation using Quokka's ``EnforceLimits`` function.

This may be necessary if the positivity-preserving timestep for a state near vacuum is too small to be feasible.
A temperature floor may also be necessary in order to prevent the auxiliary internal energy from becoming negative when there is strong cooling.
