.. Radiative shock test

Radiative shock test
====================

This test problem demonstrates the correct coupled solution of the
hydrodynamics and radiation moment equations for a subcritical
radiative shock. The steady-state solution in the nonequilibrium radiation diffusion approximation is given by a set of coupled
ODEs that can be solved to arbitrary precision following the method of :cite:`Lowrie_2008`.

Parameters
----------
The dimensionless shock parameters :cite:`Lowrie_2008` are:

.. math::
    P_0 = 1.0 \times 10^{-4} \\
    \sigma_a = 1.0 \times 10^{6} \\
    \mathcal{M}_0 = 3.0 \\
    \gamma = 5/3 \\
..

Following :cite:`Skinner_2019`, we scale to dimensional values assuming

.. math::
    \mu = m_H \\
    c_v = \frac{k_B}{\mu (\gamma - 1)} \, \text{erg} \, \text{g}^{-1} \, \text{K}^{-1} \\
    c_{s,0} = 1.73 \times 10^{7} \, \text{cm} \, \text{s}^{-1} \\
    \kappa = 577.0 \, \text{cm}^{-1} \\
..

and obtain the following pre-shock and post-shock states:

.. math::
    T_0 = 2.18 \times 10^6 \, \text{K} \\
    \rho_0 = 5.69 \, \text{g} \, \text{cm}^{-3} \\
    v_0 = 5.19 \times 10^7 \, \text{cm} \, \text{s}^{-1} \\

    T_1 = 7.98\times 10^6 \, \text{K} \\
    \rho_1 = 17.1  \, \text{g} \, \text{cm}^{-3} \\
    v_1 = 1.73 \times 10^7 \, \text{cm} \, \text{s}^{-1} \, .
..

We adopt a reduced speed of light (as used in :cite:`Skinner_2019`)

.. math::
    \hat c = 10 (v_0 + c_{s,0}) \, .
..

Solution
--------
Since the solution is given assuming radiation diffusion, we set the Eddington factor (as used in the Riemann solver for the radiation moment equations) to a constant value of :math:`1/3` everywhere.

We use the RK2 integrator with a CFL number of 0.2 and a mesh of 256 equal-size zones. After 3 shock crossing times, we obtain a solution for the radiation temperature and matter temperature that agrees to better than 0.5% (in relative L1 norm) with the steady-state ODE solution to the radiation hydrodynamics equations:


.. figure:: radshock_cgs_temperature.png
    :alt: A figure showing the radiation temperature and material temperature as a function of position.

    The radiation temperature is shown in the black solid and dashed lines, with the dashed line showing the semi-analytic solution. The material temperature is shown in the red lines, with the semi-analytic solution shown with the dashed line.