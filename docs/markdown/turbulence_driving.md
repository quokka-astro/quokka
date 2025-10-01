# Turbulence Driving Module

Quokka provides a stochastic acceleration driver that injects kinetic energy by evolving a small number of Fourier modes with an Ornstein–Uhlenbeck process. The driver builds a divergence-controlled acceleration field on the GPU and applies it after every hydrodynamic sub-step, enabling controlled Mach number targets for supersonic turbulence experiments without coupling to the hydrodynamic Riemann solver itself.【F:src/turbulence/FewModesDriver.hpp†L41-L82】【F:src/util/FewModesFT.cpp†L175-L392】

## Enabling the few-modes driver

The driver is active for every hydrodynamic problem once two conditions are met:

1. The runtime parameter block `fewmodesft` requests a non-zero forcing amplitude.
2. At least one Fourier mode is requested via `fewmodesft.num_modes`.

These checks are performed when the driver parameters are parsed, so leaving either at the default value (zero amplitude or zero modes) disables the turbulence forcing entirely.【F:src/turbulence/FewModesDriver.cpp†L53-L131】 The parameter parser runs lazily and caches its results, so repeated queries are inexpensive. The cache is cleared automatically at program finalisation and can be reset manually with `quokka::turbulence::ResetFewModesDriver()` if you launch multiple simulations within the same executable.【F:src/turbulence/FewModesDriver.cpp†L36-L125】

### Domain requirements

The phase precomputation assumes a cubic, uniformly spaced, periodic domain so that the inverse transform maps cleanly onto the mesh. `FewModesFT::SetPhases` asserts that all three physical lengths and grid dimensions are equal; the simulation aborts if these criteria are not met.【F:src/util/FewModesFT.cpp†L221-L297】 Periodic boundaries are strongly recommended so that the stochastic forcing does not create inconsistent accelerations across patch edges.

## Runtime parameters

The `fewmodesft` block accepts the following controls. Parameters omitted from the input file fall back to the defaults listed in `FewModesDriverParameters`.

- `prefix` *(default `few_modes`)* — label reserved for future diagnostics written by the driver.【F:src/turbulence/FewModesDriver.hpp†L13-L22】
- `num_modes` — number of explicit Fourier modes retained in the acceleration spectrum. Large values increase cost linearly and the constructor prints a warning above ≈100 modes.【F:src/util/FewModesFT.cpp†L175-L203】
- `k_peak` — wavenumber where the parabolic driving spectrum peaks. Random mode vectors are drawn in the range $[k_{\text{peak}}/2, 2 k_{\text{peak}}]$ and mirror symmetry is enforced so the inverse transform remains real.【F:src/util/FewModesFT.cpp†L348-L356】【F:src/util/FewModesFT.cpp†L394-L456】
- `solenoidal_weight` — fraction of power kept in the solenoidal component. Values in $[0,1]$ smoothly interpolate between purely solenoidal (1.0) and compressive (0.0) forcing; set to `-1` to skip the projection step entirely.【F:src/util/FewModesFT.cpp†L187-L360】
- `t_corr` — autocorrelation time for the Ornstein–Uhlenbeck process. Increasing `t_corr` produces more coherent forcing in time by damping the diffusion coefficient $c_\text{diff} = \sqrt{1 - e^{-2\Delta t/t_\text{corr}}}$.【F:src/util/FewModesFT.cpp†L367-L372】
- `random_seed` — base seed for both the mode selection and the per-timestep random draws. Each AMR level offsets the seed by its level index to keep the accelerations decorrelated across levels.【F:src/turbulence/FewModesDriver.cpp†L91-L118】
- `rho0`, `p0` — convenience values you can reuse while constructing initial conditions (see below). They are not consumed by the driver itself.【F:src/turbulence/FewModesDriver.cpp†L59-L99】【F:src/problems/TurbDriving/test_turb_driving.cpp†L49-L79】
- `force_amplitude` — overall scaling applied to the acceleration field before it is added to the fluid momentum. Setting this to zero cleanly disables the driver.【F:src/turbulence/FewModesDriver.cpp†L116-L118】

## Using the driver inside a problem

1. **Reset and query the parameters.** If your problem defines a custom `problem_main`, call `quokka::turbulence::ResetFewModesDriver()` before building the `QuokkaSimulation` object. The driver caches GPU resources per AMR level; resetting prevents stale allocations when running multiple problems back-to-back.【F:src/turbulence/FewModesDriver.cpp†L36-L125】 Afterwards, retrieve the parsed parameter struct with `GetFewModesDriverParameters()` whenever you need to seed initial conditions or diagnostics.【F:src/problems/TurbDriving/test_turb_driving.cpp†L49-L79】
2. **Provide compatible initial conditions.** Ensure the box dimensions and resolution satisfy the cubic constraint noted above and initialise the hydrodynamic state as usual. You can use `rho0` and `p0` from the driver parameter struct to keep the forcing amplitude and initial thermodynamics consistent across runs.【F:src/problems/TurbDriving/test_turb_driving.cpp†L49-L79】
3. **Advance the simulation.** `QuokkaSimulation` automatically invokes `ApplyFewModesDriver` at the end of every hydrodynamic level update, multiplies the stored acceleration field by the local density, and feeds the resulting momentum change back into the conserved variables. The accompanying kinetic energy correction keeps the total energy consistent with the updated momentum.【F:src/QuokkaSimulation.hpp†L735-L779】【F:src/turbulence/FewModesDriver.hpp†L41-L82】 No additional hooks are required in your time-stepping loop.

The helper `inputs/turb_driving_128.in` file demonstrates a typical configuration for a $128^3$ periodic box reaching Mach 10 turbulence. Pair it with the `test_turb_driving` problem to reproduce the gallery visualisations shown in the original pull request.【F:inputs/turb_driving_128.in†L1-L30】【F:src/problems/TurbDriving/test_turb_driving.cpp†L1-L122】

## Tips for reproducibility and tuning

- **Mode budget:** Because every mode is evolved explicitly, the GPU launch cost grows with `num_modes`. If you need a broadband spectrum, run several moderate-size mode sets with different seeds rather than one extremely large set.【F:src/util/FewModesFT.cpp†L182-L204】
- **Projection choices:** Use `solenoidal_weight = 1.0` for incompressible driving, `0.0` for compressive forcing, or intermediate values for mixed modes. The projection operates directly on the complex coefficients, so it preserves the random phases while tilting the spectrum toward your desired mixture.【F:src/util/FewModesFT.cpp†L352-L371】
- **Level-dependent forcing:** The driver automatically adds the AMR level index to the seed so that different refinement levels do not evolve the same acceleration realisation. This behaviour keeps the forcing decorrelated without further user input.【F:src/turbulence/FewModesDriver.cpp†L91-L118】
- **Reproducibility:** Provide an explicit `random_seed` and record `t_corr` and `force_amplitude` to recreate published turbulence statistics. Leave `random_seed = 0` if you want each run to draw a fresh random mode catalogue.【F:src/util/FewModesFT.cpp†L394-L456】

With these controls you can dial in driven turbulence boxes, operator-split forcing for galaxy simulations, or any other scenario requiring a controlled injection of kinetic energy.
