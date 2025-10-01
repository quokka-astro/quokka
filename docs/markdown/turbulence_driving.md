# Turbulence Driving Module

Quokka provides a stochastic acceleration driver that injects kinetic energy by evolving a small number of Fourier modes with an Ornstein–Uhlenbeck process. The driver builds a divergence-controlled acceleration field on the GPU and applies it after every hydrodynamic step ([FewModesDriver.hpp](https://github.com/quokka-astro/quokka/blob/development/src/turbulence/FewModesDriver.hpp#L41-L82), [FewModesFT.cpp](https://github.com/quokka-astro/quokka/blob/development/src/util/FewModesFT.cpp#L175-L392)).

## Enabling the few-modes driver

The driver is enabled for hydrodynamic problems if two conditions are met:

1. The runtime parameter block `fewmodesft` requests a non-zero forcing amplitude.
2. At least one Fourier mode is requested via `fewmodesft.num_modes`.

These checks are performed when the driver parameters are parsed, so leaving either at the default value (zero amplitude or zero modes) disables the turbulence forcing entirely ([FewModesDriver.cpp](https://github.com/quokka-astro/quokka/blob/development/src/turbulence/FewModesDriver.cpp#L53-L131)).

### Domain requirements

The phase precomputation assumes a cubic, uniformly spaced, periodic domain so that the inverse transform maps cleanly onto the mesh. `FewModesFT::SetPhases` asserts that all three physical lengths and grid dimensions are equal; the simulation aborts if these criteria are not met ([FewModesFT.cpp](https://github.com/quokka-astro/quokka/blob/development/src/util/FewModesFT.cpp#L221-L297)).

## Runtime parameters

The `fewmodesft` block accepts the following controls. Parameters omitted from the input file fall back to the defaults listed in `FewModesDriverParameters`.

- `num_modes` — number of explicit Fourier modes retained in the acceleration spectrum. Large values increase cost linearly and the constructor prints a warning above ≈100 modes ([FewModesFT.cpp](https://github.com/quokka-astro/quokka/blob/development/src/util/FewModesFT.cpp#L175-L203)).
- `k_peak` — wavenumber where the parabolic driving spectrum peaks. Random mode vectors are drawn in the range $[k_{\text{peak}}/2, 2 k_{\text{peak}}]$ and mirror symmetry is enforced so the inverse transform remains real ([FewModesFT.cpp](https://github.com/quokka-astro/quokka/blob/development/src/util/FewModesFT.cpp#L348-L356), [FewModesFT.cpp](https://github.com/quokka-astro/quokka/blob/development/src/util/FewModesFT.cpp#L394-L456)).
- `solenoidal_weight` — fraction of power kept in the solenoidal component. Values in $[0,1]$ smoothly interpolate between purely solenoidal (1.0) and compressive (0.0) forcing; set to `-1` to skip the projection step entirely ([FewModesFT.cpp](https://github.com/quokka-astro/quokka/blob/development/src/util/FewModesFT.cpp#L187-L360)).
- `t_corr` — autocorrelation time for the Ornstein–Uhlenbeck process. Increasing `t_corr` produces more coherent forcing in time by damping the diffusion coefficient $c_\text{diff} = \sqrt{1 - e^{-2\Delta t/t_\text{corr}}}$ ([FewModesFT.cpp](https://github.com/quokka-astro/quokka/blob/development/src/util/FewModesFT.cpp#L367-L372)).
- `random_seed` — base seed for both the mode selection and the per-timestep random draws. Each AMR level offsets the seed by its level index to keep the accelerations decorrelated across levels ([FewModesDriver.cpp](https://github.com/quokka-astro/quokka/blob/development/src/turbulence/FewModesDriver.cpp#L91-L118)).
- `force_amplitude` — overall scaling applied to the acceleration field before it is added to the fluid momentum. Setting this to zero cleanly disables the driver ([FewModesDriver.cpp](https://github.com/quokka-astro/quokka/blob/development/src/turbulence/FewModesDriver.cpp#L116-L118)).

## Using the driver inside a problem

1. **Provide compatible initial conditions.** Ensure the box dimensions and resolution satisfy the cubic constraint noted above and initialise the hydrodynamic state as usual.
2. **Advance the simulation.** `QuokkaSimulation` automatically invokes `ApplyFewModesDriver` at the end of every hydrodynamic level update, multiplies the stored acceleration field by the local density, and feeds the resulting momentum change back into the conserved variables. The accompanying kinetic energy correction keeps the total energy consistent with the updated momentum ([QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/development/src/QuokkaSimulation.hpp#L735-L779), [FewModesDriver.hpp](https://github.com/quokka-astro/quokka/blob/development/src/turbulence/FewModesDriver.hpp#L41-L82)). No additional hooks are required in your time-stepping loop.

The `inputs/turb_driving_128.in` file demonstrates a typical configuration for a $128^3$ periodic box reaching Mach 10 turbulence. Use it with the `test_turb_driving` problem to reproduce the visualisations shown in the original pull request ([turb_driving_128.in](https://github.com/quokka-astro/quokka/blob/development/inputs/turb_driving_128.in#L1-L30), [test_turb_driving.cpp](https://github.com/quokka-astro/quokka/blob/development/src/problems/TurbDriving/test_turb_driving.cpp#L1-L122)).

