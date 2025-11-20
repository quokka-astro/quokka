# Developer onboarding notes

## What Quokka is built to do
- **Radiation hydrodynamics on AMReX.** Quokka targets multi-physics astrophysical simulations using the AMReX adaptive mesh refinement framework, providing two-moment radiation transport, hydrodynamics, and optional MHD capabilities in a single-source C++20 codebase ([README.md](https://github.com/quokka-astro/quokka/blob/development/README.md#L11-L40)).
- **Modular simulation core.** The `AMRSimulation` base class orchestrates time stepping, refinement, and I/O, while `QuokkaSimulation` adds domain-specific toggles for radiation, cooling, chemistry, and MHD support that downstream problems can enable selectively ([simulation.hpp](https://github.com/quokka-astro/quokka/blob/development/src/simulation.hpp#L1-L120), [QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/development/src/QuokkaSimulation.hpp#L66-L198)).

## Repository tour
- `src/`
    - **Framework layer.** `simulation.hpp` defines `AMRSimulation`, the central driver that manages AMReX state, time stepping, outputs, and particle infrastructure ([simulation.hpp](https://github.com/quokka-astro/quokka/blob/development/src/simulation.hpp#L1-L120)).
    - **Physics modules.** Hydrodynamics, radiation, MHD, chemistry, and cooling each live in subdirectories such as `hydro/`, `radiation/`, and `cooling/`, which are wired into `QuokkaSimulation` via the `Physics_Traits` mechanism ([QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/development/src/QuokkaSimulation.hpp#L66-L200)).
    - **Problem drivers.** Each scenario in `src/problems/` defines a `problem_t` type, customises traits, sets initial conditions, and then instantiates `QuokkaSimulation` to run; see [`src/problems/OrszagTang/testOrszagTang.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/OrszagTang/testOrszagTang.cpp#L27-L138) for a representative setup.
- `inputs/`
    - Runtime parameter files that pair with problem drivers; regression entries reference them directly when defining automated tests ([regression/quokka-tests.ini](https://github.com/quokka-astro/quokka/blob/development/regression/quokka-tests.ini#L65-L146)).
- `regression/`
    - The regression harness (`quokka-tests.ini`) enumerates long-running GPU test suites, including MPI launch commands, linked data files, and which executables to build ([regression/quokka-tests.ini](https://github.com/quokka-astro/quokka/blob/development/regression/quokka-tests.ini#L1-L146)).
- `docs/`
    - Source for the published documentation site (MkDocs). The landing page summarises Quokka’s goals and AMReX integration, and additional pages cover workflow diagrams, testing, debugging, and performance topics ([site overview](index.md), [simulation flowchart](flowchart.md), [test catalog](tests/index.md)).

## Execution flow in practice
- Start-up: `main.cpp` initialises AMReX, then calls `problem_main()` declared in `main.hpp` and implemented by each problem driver ([main.cpp](https://github.com/quokka-astro/quokka/blob/development/src/main.cpp#L1-L55), [main.hpp](https://github.com/quokka-astro/quokka/blob/development/src/main.hpp#L8-L19)).
- Simulation lifecycle: the flowchart documentation page mirrors the control flow of `AMRSimulation::setInitialConditions`, `evolve`, and `computeTimestep`, showing the nested loops for hydrodynamics stages and radiation subcycling ([flowchart overview](flowchart.md)).
- Custom physics: problem-specific traits determine which subsystems `QuokkaSimulation` activates (e.g., MHD, radiation groups, chemistry), and each problem supplies initial conditions (cells and face-centered fields) before calling `sim.evolve()` ([QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/development/src/QuokkaSimulation.hpp#L183-L199), [`testOrszagTang.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/OrszagTang/testOrszagTang.cpp#L30-L138)).

## Build, test, and quality checks
- **Build locally.** Follow the installation guide to clone with submodules, configure with CMake (Ninja or Make), and choose the desired dimensionality and accelerator backend ([installation guide](installation.md)).
- **Automated tests.** `ninja test` or `ctest` exercises the bundled problem suite; for full GPU coverage, rely on the regression harness described earlier ([installation guide](installation.md), [quokka-tests.ini](https://github.com/quokka-astro/quokka/blob/development/regression/quokka-tests.ini#L65-L146)).
- **Static analysis.** Run `clang-tidy` manually or via `scripts/tidy.sh` to match the repository’s CI checks, as documented in the How to Use clang-tidy guide ([clang-tidy how-to](howto_clang_tidy.md)).
- **CUDA builds on macOS or Linux.** To build and test CUDA functionality locally, run `./scripts/bash/run-cuda-container.sh`. This script pulls the appropriate Docker image, launches a container, and performs a CUDA build—useful for catching CUDA-specific issues on your development machine.

## Where to dive deeper next
1. **Physics modules.** Explore `src/hydro/` and `src/radiation/` alongside the corresponding documentation pages (`hydro_integrator.md`, radiation topics) to understand scheme implementations ([QuokkaSimulation.hpp](https://github.com/quokka-astro/quokka/blob/development/src/QuokkaSimulation.hpp#L66-L200), [overview page](index.md)).
2. **Problem setups.** Study more drivers in `src/problems/` and the matching documentation in the Tests section to see how diagnostic comparisons are scripted ([`testOrszagTang.cpp`](https://github.com/quokka-astro/quokka/blob/development/src/problems/OrszagTang/testOrszagTang.cpp#L27-L138), [test index](tests/index.md)).
3. **Performance & HPC workflows.** The GPU section in the installation guide plus the Running on HPC Clusters page outline how to scale to accelerators and clusters ([installation guide](installation.md#running-on-gpus), [HPC guide](running_on_hpc_clusters.md)).
4. **Contribution process.** Pair the coding standards baked into `clang-tidy` with the community guidance in the Contributing guide when preparing patches (and review `CONTRIBUTING.md` at the repo root) ([clang-tidy how-to](howto_clang_tidy.md), [contribution guide](contributing.md)).

Armed with the overview above, a newcomer can pick a problem driver, follow the build instructions, and iterate confidently while leaning on the documentation site for deeper dives.
