# Developing a new problem generator

Quokka problem generators live under `src/problems/`, each in its own subdirectory with a driver source file and a small CMake fragment. The sections below walk through the essential steps to bring up a new scenario, from adding the entry point to integrating it with the build and test infrastructure.

## 1. Create a problem skeleton

1. Pick a descriptive directory name (for example `MyProblem`) beneath `src/problems/` and add an `add_subdirectory(MyProblem)` entry to `src/problems/CMakeLists.txt`. This is how the top-level build discovers the new problem; see [`src/problems/CMakeLists.txt`](https://github.com/quokka-astro/quokka/blob/development/src/problems/CMakeLists.txt#L1-L71).
2. Inside the new directory, create a `CMakeLists.txt` that uses the `quokka_add_problem` helper function from `ProblemHelpers.cmake`. This function automatically sets up the executable (with the correct sources and CUDA compilation if needed) and optionally registers a regression test. The simplest case is:
   ```cmake
   quokka_add_problem(JOB_NAME MyProblem)
   ```
   This creates an executable named `MyProblem` from `testMyProblem.cpp`, sets up CUDA compilation if needed, and registers a test that runs with `../inputs/MyProblem.toml`. To disable the test, use `ADD_TEST OFF`:
   ```cmake
   quokka_add_problem(JOB_NAME MyProblem ADD_TEST OFF)
   ```
   For problems that need dimension guards or custom input files, you can combine the helper with conditional logic:
   ```cmake
   if(AMReX_SPACEDIM GREATER_EQUAL 3)
     quokka_add_problem(JOB_NAME MyProblem)
   endif()
   ```
   See [`src/problems/ShockCloud/CMakeLists.txt`](https://github.com/quokka-astro/quokka/blob/development/src/problems/ShockCloud/CMakeLists.txt) and [`src/problems/NscbcVortex/CMakeLists.txt`](https://github.com/quokka-astro/quokka/blob/development/src/problems/NscbcVortex/CMakeLists.txt) for examples. The helper function's full interface is documented in [`src/problems/ProblemHelpers.cmake`](https://github.com/quokka-astro/quokka/blob/development/src/problems/ProblemHelpers.cmake). For problems requiring multiple tests or more complex test configurations, you may need to manually set up the executable and tests instead of using the helper. See [`src/problems/testSN/CMakeLists.txt`](https://github.com/quokka-astro/quokka/blob/development/src/problems/testSN/CMakeLists.txt) for an example of a problem that requires multiple tests.
3. Add a driver source file (for example `testMyProblem.cpp`) that will hold the problem-specific specialisations and the `problem_main()` implementation described below.

## 2. Define problem traits

Problem generators tag their configuration by declaring an empty type (e.g., `struct MyProblem { };`) and specialising the trait structures that drive Quokka’s compile-time selection of physics modules.

* Every problem must specialise `Physics_Traits<MyProblem>` to advertise which subsystems (hydrodynamics, radiation, MHD, etc.) are active and which unit system is in use. The linear advection driver demonstrates a minimal specialisation that disables all optional physics; examine [`src/problems/Advection/test_advection.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/Advection/test_advection.cpp#L30-L43).
* Problems that rely on an equation of state should also specialise `quokka::EOS_Traits<MyProblem>` to provide constants such as the adiabatic index and mean molecular weight; the hydro shock tube example illustrates the pattern in [`src/problems/HydroShocktube/test_hydro_shocktube.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/HydroShocktube/test_hydro_shocktube.cpp#L33-L52).

Once the traits are in place you can specialise the Quokka or AMR simulation hooks that actually set up and evolve your scenario.

## 3. Provide initial conditions and runtime hooks

At a minimum, implement `setInitialConditionsOnGrid` for your problem’s simulation type. This routine is invoked during `setInitialConditions()` and must populate the cell-centred state arrays for each patch. The advection test fills a sawtooth profile by looping over the grid supplied through the helper `quokka::grid` struct; refer to [`src/problems/Advection/test_advection.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/Advection/test_advection.cpp#L56-L68).

Additional hooks are available when you need them:

* Override `setCustomBoundaryConditions` if you require non-periodic inflow/outflow values, as shown by the shock tube problem described in [`src/problems/HydroShocktube/test_hydro_shocktube.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/HydroShocktube/test_hydro_shocktube.cpp#L102-L152).
* Provide a `refineGrid` specialisation to tag cells for AMR refinement. The shock tube driver flags zones based on the density gradient magnitude—see [`src/problems/HydroShocktube/test_hydro_shocktube.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/HydroShocktube/test_hydro_shocktube.cpp#L154-L177).
* Implement `computeReferenceSolution` when you want the regression harness to compare against an analytic or tabulated solution. The advection example computes an exact profile for error checking and optional plotting in [`src/problems/Advection/test_advection.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/Advection/test_advection.cpp#L70-L122).

Many other virtual hooks (for particles, diagnostics, derived variables, etc.) already have defaults in `QuokkaSimulation`, so you only need to specialise the ones your problem truly depends on.

### Boundary conditions

Boundaries of type `ext_dir` in `quokka.bc` are filled by the problem's `setCustomBoundaryConditions` (cell-centred) and `setCustomBoundaryConditionsFaceVar` (face-centred) specialisations.

#### Diode (outflow, no-inflow)

The diode lets gas leave the domain and stops it from entering. For hydro, call `setDiodeBCLo<dir>` / `setDiodeBCHi<dir>` in `setCustomBoundaryConditions`. For MHD, additionally set the face-centred boundary to `foextrap` (a placeholder) and specialise `AMRSimulation<problem_t>::isMHDDiodeBoundary(dir, side)` to return `true`; `applyMHDDiodeBC` then fills the ghost magnetic field after the regular ghost fills. `src/problems/MHDDiode` is an example.

We take the lower \\(x\\) boundary as the example: \\(x\\) is the normal direction and \\(y\\), \\(z\\) are transverse; the other boundaries follow by symmetry. Cell \\(0\\) is the first valid cell, ghost cell \\(-m\\) has left face \\(-m\\), and face \\(0\\) is the boundary face. Each column (fixed \\(j\\), \\(k\\)) is outflow if \\(\rho v\_x < 0\\) in its first valid cell and inflow otherwise.

| Quantity | Outflow column | Inflow column |
|---|---|---|
| \\(\rho\\), \\(\rho v\_y\\), \\(\rho v\_z\\), internal energy, scalars | copy of cell \\(0\\) | mirror of cell \\(m-1\\) |
| \\(\rho v\_x\\) | copy of cell \\(0\\) | mirror of cell \\(m-1\\), sign reversed |
| \\(B\_y\\), \\(B\_z\\) on ghost faces | copy of cell \\(0\\) | mirror of cell \\(m-1\\), same sign |
| \\(B\_x\\) on ghost faces | from \\(\nabla \cdot \mathbf{B} = 0\\), eq. (1) | from \\(\nabla \cdot \mathbf{B} = 0\\), eq. (1) |
| \\(B\_x\\) on the boundary face | not written (evolved by CT) | not written (evolved by CT) |
| total energy | eq. (2) with source cell \\(0\\) | eq. (2) with source cell \\(m-1\\) |

A transverse ghost face shared by an inflow and an outflow column is treated as inflow. The normal field and the total energy in the ghost cells are

<script type="math/tex; mode=display">
B_x(-m) = B_x(-m+1) + \Delta x \left[ \frac{B_y(-m, j+1) - B_y(-m, j)}{\Delta y} + \frac{B_z(-m, k+1) - B_z(-m, k)}{\Delta z} \right], \qquad m = 1, \dots, n_g, \qquad (1)
</script>

<script type="math/tex; mode=display">
E_{\rm tot}^{\rm ghost} = E_{\rm tot}^{\rm source} - E_B^{\rm source} + E_B^{\rm ghost}, \qquad (2)
</script>

with \\(E\_B\\) from the face-averaged field. The source cell is the valid cell whose state the cell-centred fill copied into the ghost cell: cell \\(0\\) (outflow) or cell \\(m-1\\) (inflow). Since the kinetic energies of the ghost and source cells are equal, eq. (2) is the same as \\(E\_{\rm tot}^{\rm ghost} = E\_{\rm int}^{\rm source} + E\_{\rm kin}^{\rm ghost} + E\_B^{\rm ghost}\\) with \\(E\_{\rm int}^{\rm source} = E\_{\rm tot}^{\rm source} - E\_{\rm kin}^{\rm source} - E\_B^{\rm source}\\), the internal energy from which `ComputePrimVars` computes the pressure (not the auxiliary internal energy). In a fully mirrored column, eq. (1) gives \\(B\_x(-m) = 2 B\_x(0) - B\_x(m)\\).

Justification:

- **Normal field from the divergence constraint.** Eq. (1) makes every ghost cell divergence-free for any transverse values, so the inflow/outflow pattern, and a column switching type, cannot create \\(\nabla \cdot \mathbf{B}\\). The boundary face is updated by CT with a curl, so no EMF boundary condition is needed (unlike the EMF-based diode of [@Pjanka2020]).
- **No conducting-wall reflection.** Reflecting \\(B\_x\\) as an odd function, \\(B\_x(-m) = -B\_x(m)\\), is divergence-free only if \\(B\_x(0) = 0\\); otherwise the first ghost cell has \\(\nabla \cdot \mathbf{B} = 2 B\_x(0) / \Delta x\\).
- **Mirror, not copy-and-flip, at inflow.** With reconstruction order above one, the face states at the boundary are mirror images only if the ghost data are a geometric mirror. Copying cell \\(0\\) and flipping the momentum loses a fraction \\(2.7 \times 10^{-3}\\) of the mass in the `MHDDiode` wall test.
- **Zero mass flux.** For mirror-image face states (equal \\(\rho\\), \\(P\\), \\(B\_y^2 + B\_z^2\\), opposite \\(v\_x\\)), the numerator of the HLLD middle speed (eq. 38 of [@Miyoshi2005]), \\((S\_R - v\_{x,R})\rho\_R v\_{x,R} - (S\_L - v\_{x,L})\rho\_L v\_{x,L} + p\_{T,L} - p\_{T,R}\\), is exactly zero, so \\(S\_M = 0\\) and the mass flux vanishes to round-off.
- **Energy correction.** The copied total energy contains the source-cell magnetic energy, while the ghost field differs. Without eq. (2) the ghost pressure is wrong, the face states are not mirror images, and mass leaks (\\(3 \times 10^{-6}\\) with PLM, \\(10^{-5}\\) with xPPM in the wall test; PPM hides the error because it falls to first order at the wall).
- **Same-sign mirror of \\(B\_y\\), \\(B\_z\\).** Mirroring them with the opposite sign (\\(B\_x\\) even, \\(B\_y\\), \\(B\_z\\) odd) is also divergence-free and also gives \\(S\_M = 0\\), but it puts a current sheet at the wall and flips the ghost \\(B\_y\\), \\(B\_z\\) every time a column switches, which jumps the boundary-edge EMFs.
- **Shared faces as inflow.** This keeps the ghost data of every inflow column exactly mirrored, which zero mass flux needs; it only slightly changes the neighbouring outflow column.

**Limitation.** The diode constrains the mass flux only. Where \\(B\_x(0) \neq 0\\), tangential flow at the wall gives a non-zero boundary EMF, so magnetic flux can enter through inflow faces, and the wall can carry magnetic stress and Poynting flux.

The `MHDDiode` tests check zero mass flux with an all-inflow boundary, a non-increasing mass with mixed inflow and outflow, and \\(\nabla \cdot \mathbf{B} = 0\\) to round-off in all valid and ghost cells.

## 4. Write `problem_main()`

The `problem_main()` function is the entry point that `src/main.cpp` calls after AMReX initialisation. It belongs in your driver file and should construct the appropriate simulation class, configure runtime parameters, and launch the run. The advection driver shows the typical flow: build boundary conditions, instantiate the simulation, adjust stopping criteria and CFL numbers, seed the initial data, and finally call `evolve()`—compare [`src/problems/Advection/test_advection.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/Advection/test_advection.cpp#L124-L158) and [`src/main.hpp`](https://github.com/quokka-astro/quokka/blob/development/src/main.hpp#L16-L18).

If your problem needs exact solutions, diagnostics, or error checks, compute them before returning a status code from `problem_main()` so automated tests can detect failures.

## 5. Build and run the problem

1. Regenerate or update your build tree with CMake (for example, `cmake -S . -B build -G Ninja` with the desired options). The compiled problem executables live under `build/src/problems/<ProblemName>/` once the build completes; the installation guide covers the workflow in the [build instructions](installation.md#installation).
2. Ask CMake for the target corresponding to your new problem (e.g., `cmake --build build --target help`) and then build it with Ninja or your chosen generator. The executable name matches the `JOB_NAME` you specified in `quokka_add_problem` (e.g., `MyProblem`), as outlined in the [specific-target build section](installation.md#building-a-specific-test-problem).
3. Run the binary from a working directory that can see your input deck, passing the `.toml` file path as the first argument. The regression tests invoke the advection example as `Advection ../inputs/Advection.toml` (note the executable name matches the `JOB_NAME`), which you can mimic for manual runs. The `quokka_add_problem` helper automatically configures tests to run from the `tests/` directory with the input file path `../inputs/${JOB_NAME}.toml`.

Following the steps above should give you a fully integrated problem generator that participates in Quokka’s build system and can be exercised both manually and through CTest.
