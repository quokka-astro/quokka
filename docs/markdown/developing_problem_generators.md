# Developing a new problem generator

Quokka problem generators live under `src/problems/`, each in its own subdirectory with a driver source file and a small CMake fragment. The sections below walk through the essential steps to bring up a new scenario, from adding the entry point to integrating it with the build and test infrastructure.

## 1. Create a problem skeleton

1. Pick a descriptive directory name (for example `MyProblem`) beneath `src/problems/` and add an `add_subdirectory(MyProblem)` entry to `src/problems/CMakeLists.txt`. This is how the top-level build discovers the new problem; see [`src/problems/CMakeLists.txt`](https://github.com/quokka-astro/quokka/blob/development/src/problems/CMakeLists.txt#L1-L71).
2. Inside the new directory, create a `CMakeLists.txt` that uses the `quokka_add_problem` helper function from `ProblemHelpers.cmake`. This function automatically sets up the executable (with the correct sources and CUDA compilation if needed) and optionally registers a regression test. The simplest case is:
   ```cmake
   quokka_add_problem(JOB_NAME MyProblem)
   ```
   This creates an executable named `MyProblem` from `testMyProblem.cpp`, sets up CUDA compilation if needed, and registers a test that runs with `../inputs/MyProblem.in`. To disable the test, use `ADD_TEST OFF`:
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

## 4. Write `problem_main()`

The `problem_main()` function is the entry point that `src/main.cpp` calls after AMReX initialisation. It belongs in your driver file and should construct the appropriate simulation class, configure runtime parameters, and launch the run. The advection driver shows the typical flow: build boundary conditions, instantiate the simulation, adjust stopping criteria and CFL numbers, seed the initial data, and finally call `evolve()`—compare [`src/problems/Advection/test_advection.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/Advection/test_advection.cpp#L124-L158) and [`src/main.hpp`](https://github.com/quokka-astro/quokka/blob/development/src/main.hpp#L16-L18).

If your problem needs exact solutions, diagnostics, or error checks, compute them before returning a status code from `problem_main()` so automated tests can detect failures.

## 5. Build and run the problem

1. Regenerate or update your build tree with CMake (for example, `cmake -S . -B build -G Ninja` with the desired options). The compiled problem executables live under `build/src/problems/<ProblemName>/` once the build completes; the installation guide covers the workflow in the [build instructions](installation.md#installation).
2. Ask CMake for the target corresponding to your new problem (e.g., `cmake --build build --target help`) and then build it with Ninja or your chosen generator. The executable name matches the `JOB_NAME` you specified in `quokka_add_problem` (e.g., `MyProblem`), as outlined in the [specific-target build section](installation.md#building-a-specific-test-problem).
3. Run the binary from a working directory that can see your input deck, passing the `.in` file path as the first argument. The regression tests invoke the advection example as `Advection ../inputs/Advection.in` (note the executable name matches the `JOB_NAME`), which you can mimic for manual runs. The `quokka_add_problem` helper automatically configures tests to run from the `tests/` directory with the input file path `../inputs/${JOB_NAME}.in`.

Following the steps above should give you a fully integrated problem generator that participates in Quokka’s build system and can be exercised both manually and through CTest.
