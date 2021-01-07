.. Slow-moving shock test

Slow-moving shock test
=======================

This test problem demonstrates the extent to which post-shock oscillations
are controlled in slowly-moving shocks. This effect can be exhibited in all
Godunov codes, even with first-order methods, for sufficiently slow-moving
shocks across the computational grid :cite:`Jin_1996`.

The shock flattening method of :cite:`CW84` (implemented in our code in modified form) reduces the oscillations, but does not completely
suppress them. Adding artificial viscosity according to the method of :cite:`CW84`,
even to the level of smoothing the contact discontinuity by 5-10 cells, does `not` cure the problem.

Parameters
----------
The left- and right-side initial conditions are :cite:`Quirk_1994`:

.. math::
	\rho_0 = 3.86 \\
	v_{x,0} = -0.81 \\
	P_0 = 10.3334 \\

	\rho_0 = 1.0 \\
	v_{x,0} = -3.44 \\
	P_0 = 1.0
..

The shock moves to the right with speed :math:`s = 0.1096`.

Solution
--------

We use the RK2 integrator with a fixed timestep of :math:`10^{-3}`
and a mesh of 100 equally-spaced cells. The contact discontinuity
is initially placed at :math:`x=0.5`.

.. figure:: hydro_sms.png
    :alt: A figure showing the numerical solution to the slow-moving shock test.

    The density is shown as the solid blue line. The exact solution is the solid orange line.