# Developer onboarding notes

## What Quokka is built to do
- **Radiation hydrodynamics on AMReX.** Quokka targets multi-physics astrophysical simulations using the AMReX adaptive mesh refinement framework, providing two-moment radiation transport, hydrodynamics, and optional MHD capabilities in a single-source C++17 codebase ([README.md](https://github.com/quokka-astro/quokka/blob/main/README.md#L11-L40)).
- **Modular simulation core.** The `AMRSimulation` base class orchestrates time stepping, refinement, and I/O, while `QuokkaSimulation` adds domain-specific toggles for radiation, cooling, chemistry, and MHD support that downstream problems can enable selectively ([simulation.hpp](https://github.com/quokka-astro/quokka/blob/main/src/simulation.hpp#L1-L120), [QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/main/src/QuokkaSimulation.hpp#L66-L198)).

## Repository tour
- `src/`
  - **Framework layer.** `simulation.hpp` defines `AMRSimulation`, the central driver that manages AMReX state, time stepping, outputs, and particle infrastructure ([simulation.hpp](https://github.com/quokka-astro/quokka/blob/main/src/simulation.hpp#L1-L120)).
  - **Physics modules.** Hydrodynamics, radiation, MHD, chemistry, and cooling each live in subdirectories such as `hydro/`, `radiation/`, and `cooling/`, which are wired into `QuokkaSimulation` via the `Physics_Traits` mechanism ([QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/main/src/QuokkaSimulation.hpp#L66-L200)).
  - **Problem drivers.** Each scenario in `src/problems/` defines a `problem_t` type, customises traits, sets initial conditions, and then instantiates `QuokkaSimulation` to run; see [`src/problems/OrszagTang/test_orszag_tang.cpp`](https://github.com/quokka-astro/quokka/blob/main/src/problems/OrszagTang/test_orszag_tang.cpp#L27-L138) for a representative setup.
- `inputs/`
  - Runtime parameter files that pair with problem drivers; regression entries reference them directly when defining automated tests ([regression/quokka-tests.ini](https://github.com/quokka-astro/quokka/blob/main/regression/quokka-tests.ini#L65-L146)).
- `regression/`
  - The regression harness (`quokka-tests.ini`) enumerates long-running GPU test suites, including MPI launch commands, linked data files, and which executables to build ([regression/quokka-tests.ini](https://github.com/quokka-astro/quokka/blob/main/regression/quokka-tests.ini#L1-L146)).
- `docs/`
  - Source for the published documentation site (MkDocs). The landing page summarises Quokka’s goals and AMReX integration, and additional pages cover workflow diagrams, testing, debugging, and performance topics ([index.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/index.md#L1-L18), [flowchart.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/flowchart.md#L1-L84), [tests/index.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/tests/index.md#L1-L10)).

## Execution flow in practice
- Start-up: `main.cpp` initialises AMReX, then calls `problem_main()` declared in `main.hpp` and implemented by each problem driver ([main.cpp](https://github.com/quokka-astro/quokka/blob/main/src/main.cpp#L1-L55), [main.hpp](https://github.com/quokka-astro/quokka/blob/main/src/main.hpp#L8-L19)).
- Simulation lifecycle: the flowchart in `docs/markdown/flowchart.md` mirrors the control flow of `AMRSimulation::setInitialConditions`, `evolve`, and `computeTimestep`, showing the nested loops for hydrodynamics stages and radiation subcycling ([flowchart.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/flowchart.md#L12-L84)).
- Custom physics: problem-specific traits determine which subsystems `QuokkaSimulation` activates (e.g., MHD, radiation groups, chemistry), and each problem supplies initial conditions (cells and face-centered fields) before calling `sim.evolve()` ([QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/main/src/QuokkaSimulation.hpp#L183-L199), [`test_orszag_tang.cpp`](https://github.com/quokka-astro/quokka/blob/main/src/problems/OrszagTang/test_orszag_tang.cpp#L30-L138)).

## Build, test, and quality checks
- **Build locally.** Follow the installation guide to clone with submodules, configure with CMake (Ninja or Make), and choose the desired dimensionality and accelerator backend ([installation.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/installation.md#L1-L106)).
- **Automated tests.** `ninja test` or `ctest` exercises the bundled problem suite; for full GPU coverage, rely on the regression harness described earlier ([installation.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/installation.md#L18-L33), [quokka-tests.ini](https://github.com/quokka-astro/quokka/blob/main/regression/quokka-tests.ini#L65-L146)).
- **Static analysis.** Run `clang-tidy` manually or via `scripts/tidy.sh` to match the repository’s CI checks, as documented in `docs/markdown/howto_clang_tidy.md` ([howto_clang_tidy.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/howto_clang_tidy.md#L1-L45)).

## Where to dive deeper next
1. **Physics modules.** Explore `src/hydro/` and `src/radiation/` alongside the corresponding documentation pages (`hydro_integrator.md`, `radiation` topics) to understand scheme implementations ([QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/main/src/QuokkaSimulation.hpp#L66-L200), [index.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/index.md#L12-L18)).
2. **Problem setups.** Study more drivers in `src/problems/` and the matching documentation under `docs/markdown/tests/` to see how diagnostic comparisons are scripted ([`test_orszag_tang.cpp`](https://github.com/quokka-astro/quokka/blob/main/src/problems/OrszagTang/test_orszag_tang.cpp#L27-L138), [tests/index.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/tests/index.md#L1-L10)).
3. **Performance & HPC workflows.** The `installation.md` GPU section plus `running_on_hpc_clusters.md` (in the same docs tree) outline how to scale to accelerators and clusters ([installation.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/installation.md#L65-L106), [running_on_hpc_clusters.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/running_on_hpc_clusters.md#L1-L33)).
4. **Contribution process.** Pair the coding standards baked into `clang-tidy` with the community guidance in `docs/markdown/contributing.md` when preparing patches (and review `CONTRIBUTING.md` at the repo root) ([howto_clang_tidy.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/howto_clang_tidy.md#L1-L45), [contributing.md](https://github.com/quokka-astro/quokka/blob/main/docs/markdown/contributing.md#L1-L138)).

Armed with the overview above, a newcomer can pick a problem driver, follow the build instructions, and iterate confidently while leaning on the documentation site for deeper dives.
