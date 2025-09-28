# Star Formation

## Overview

The star formation module adds star particles through a lightweight stochastic prescription that plugs into the Stochastic Stellar Population specialisation.

- Runs once per hydro timestep and evaluates each cell independently.
- Always spawns a low-mass star particle when a trial succeeds.
- Adds high-mass particles probabilistically so the Chabrier initial mass function (IMF) is satisfied in expectation.

## Jeans instability filter

Eligible cells are first identified through a Jeans-length check before any stochastic sampling occurs.

- Compute the Jeans length \(\lambda_J = c_s / \sqrt{G \rho}\) in every cell.
- Mark the cell as eligible when \(\lambda_J < J \cdot \Delta x\) with \(J = 0.5\).
- Only eligible cells continue to the sampling steps below.

## Controlling the formation rate

Two efficiency parameters tune how aggressively eligible gas is converted into star particles during each hydro step.

- \(\epsilon_{ff}\): efficiency per free-fall time, defined through \(\epsilon_{ff} = (\dot M_{\star} \, t_{ff}) / (M_{cell} \, \Delta t)\).
- \(\epsilon_{\star}\): fraction of the cell mass used when a particle is spawned.
- The target stellar mass for the step is \(\epsilon_{ff} M_{cell} (\Delta t / t_{ff})\).
- Bernoulli probability for spawning: \( P = \frac{\epsilon_{ff}}{\epsilon_{\star}} \frac{\Delta t}{t_{ff}} \).
- The expectation value \(\langle M_{\star} \rangle = P \epsilon_{\star} M_{cell}\) matches the target mass provided \(\Delta t < t_{ff}\); the CFL condition typically enforces that inequality.

## Sampling the stellar population

Once a cell passes the filter and the Bernoulli draw succeeds, we construct the composite stellar population represented by the spawned particles.

- Every accepted draw creates one low-mass particle that represents all stars with \(M < 8 M_{\odot}\).
- High-mass stars follow the Chabrier (2003) IMF: log-normal below \(1 M_{\odot}\) and a slope of \(2.35\) above it.
- Pre-computed IMF integrals provide the mass fraction \(f_{\star,high}\) and mean mass \(\langle m \rangle_{\star,high}\) of the high-mass component.
- The expected number of massive stars is \(f_{\star,high} \epsilon_{\star} M_{cell} / \langle m \rangle_{\star,high}\); a Poisson variate with that mean sets the actual count.
- Each massive star draws its mass from the high-mass end of the IMF, while the low-mass particle retains the remaining fraction \(1 - f_{\star,high}\) of the spawned mass.
- If the Poisson draw returns zero, only the low-mass particle is inserted and it inherits the local gas velocity.

## Assigning particle velocities

Sampled star particles receive velocities that combine the local bulk flow with an isotropic runaway kick.

- Each massive star draws a speed from a power-law distribution \(p(v) \propto v^{-1.8}\) truncated between 3 and 385 km s\(^{-1}\), then converts it to cgs units.
- A random direction is chosen by sampling \(\cos \theta\) uniformly in \([-1, 1]\) and \(\phi\) in \([0, 2\pi)\); the resulting kick is added to the gas velocity of the parent cell.
- The total momentum of all massive stars is accumulated, and the low-mass composite particle receives the opposite momentum so that the cell-level particle system conserves momentum.

## Practical considerations

A few implementation notes help interpret corner cases and limitations of the current recipe.

- Star formation is operator-split from the hydrodynamics. Large timesteps compared with \(t_{ff}\) will therefore overshoot the desired rate; no explicit limiter is enforced beyond the CFL-controlled hydro step.
- All spawned particles are inserted at the cell centre. Other physics modules are responsible for any subsequent repositioning or feedback coupling.

