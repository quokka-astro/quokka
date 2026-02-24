---
name: quokka-dev
description: "Implement and validate new features in the Quokka radiation hydrodynamics codebase. Use when: (1) Adding new feature, (2) Adding new physics problems or test cases, (3) Modifying core simulation code in hydro/radiation/cooling/chemistry/particle/gravity modules. IMPORTANT: Always validate implementation by compiling and running a test problem. Use the SN test as default if no specific test is specified."
---
# Quokka Development Workflow


## Implementation Workflow

### 1. Understand the Feature

- Identify which module the feature belongs to: `src/hydro/`, `src/radiation/`, `src/cooling/`, `src/chemistry/`, or a new problem in `src/problems/`
- Review existing similar implementations for patterns

### 2. Create or Modify Code

**For new problems:**

- Create directory `src/problems/<TestName>/`
- Create `test<TestName>.cpp` with:
  - Template specialization for `QuokkaSimulation<ProblemName>`
  - Initial conditions via `setInitialConditionsOnGrid`
  - Problem-specific physics callbacks as needed
- Create `CMakeLists.txt` defining the executable target
- Create input file `inputs/<TestName>.in`

**For core module changes:**

- Modify files in the appropriate `src/` subdirectory
- Follow the `QuokkaSimulation` / `AMRSimulation` inheritance pattern

**For I/O changes (`src/io/`):**

- DiagPlotfile is the current plotfile diagnostic system, configured via `quokka.plt.*` input parameters. Some older tests still use the legacy `plotfile_interval` parameter instead.
- Plotfile output goes to `<REPO_ROOT>/tests/plt<step>/` (e.g. `<REPO_ROOT>/tests/plt0000006/`)
- After running, verify the output by checking:
  - `tests/plt<step>/Header` — lists variable names and count
  - `tests/plt<step>/fc_vars/` — subdirectory for face-centered data (MHD only), with per-direction sub-subdirectories (`x<step>/`, `y<step>/`, `z<step>/`)
  - Each `fc_vars/<dir><step>/Header` lists the variable(s) for that direction

### 3. Code Style Requirements

- 160 character line limit, 8-space tabs
- PascalCase for classes and member functions
- camelCase with trailing underscore for member variables
- Always use curly braces for single statement blocks
- Always use trailing return type for non-void functions
- Declare variables `const` when not modified after initialization
- Format with `.clang-format` from `src/`

### 4. Build and Test

**At the start of every session, read the repo root from `pwd`** — Claude is always launched from the repo root, which may be any worktree (e.g. `quokka`, `quokka2`, `quokka3`). Use `$(pwd)` or run `pwd` to get it, then derive all paths from it:

```
<REPO_ROOT>/           ← pwd at session start
├── build/clang-3d/    ← build directory
├── tests/             ← test runner (Makefile lives here)
├── src/
└── inputs/
```

All build and run commands must be invoked from `<REPO_ROOT>/tests/`. Load modules and change directory once per shell session (substitute the actual `<REPO_ROOT>` path):

```bash
source ~/rc/qk.rc && cd <REPO_ROOT>/tests
```

Then build:

```bash
JOB=<TestName> make b
```

Run the test:

```bash
JOB=<TestName> make r
```

Or, if implementing MPI-related features, run the test with MPI:

```bash
JOB=<TestName> make rmpi
```

Plotfile output lands in `<REPO_ROOT>/tests/` (e.g. `<REPO_ROOT>/tests/plt0000006/`).

### 5. Language Server Diagnostics

The language server shows many false-positive errors (`Use of undeclared identifier 'amrex'`, `file not found`, etc.) because AMReX headers are not in its include path. **Ignore these** — they are not real compiler errors. Only trust actual build output from `make b`.

### 6. Validate

**Always validate your implementation by compiling and running a test problem.**

- If working on a specific problem, build and run that problem
- If no specific problem is specified, use the **SN** test as the default validation target
- Ensure build completes without errors
- Ensure test passes
- Check for compiler warnings


## Architecture Reference

| Component          | Location                 | Purpose                                     |
| ------------------ | ------------------------ | ------------------------------------------- |
| Main entry         | `src/main.cpp`         | Calls `problem_main()`                    |
| Core simulation    | `QuokkaSimulation`     | Template class inheriting `AMRSimulation` |
| Hyperbolic systems | `HyperbolicSystem`     | Conservation laws, slope limiters           |
| Problems           | `src/problems/<Name>/` | Problem generator                           |
| I/O                | `src/io/`              | Plotfiles, checkpoints, openPMD             |
| Math utilities     | `src/math/`            | Interpolation, quadrature, ODE              |
| Particles          | `src/particles/`       | Particles                                   |

## Documentation

Create a PR.md file in the root directory to document the changes. Keep it short and concise.

In the end of the implementation, summarize what's been changed in a few sentences in the chat window.

## Create Pull Request

When the feature implementation is complete, ask the user for confirmation before opening a PR. Use the GitHub CLI (gh) to open a draft pull request from the current branch with the description read from PR.md.

1. Ensure the branch is pushed. From the feature branch:

  git push -u origin HEAD

2. Create the draft PR

  gh pr create --base development --title "A descriptive title" --body-file PR.md --draft

- `--body-file PR.md` → uses the Markdown file as the PR description
- `--draft` → opens as a Draft PR
- The PR URL is printed to stdout
