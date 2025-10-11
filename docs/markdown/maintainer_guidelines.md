# Maintainers Guide

Quokka maintainers safeguard long-term technical direction and scientific integrity for a multi-physics AMReX application that must build across multiple toolchains and hardware platforms. You review pull requests (PRs = proposed code changes) to ensure they are **correct**, **efficient**, **portable** (CPU/GPU; CUDA or HIP; x86 or ARM; macOS or Linux), and **reproducible** (others can repeat results).

!!! important "Stability expectations"
    - **Problem-file interface** may change occasionally to support new features.
    - **Runtime `ParmParse` options** should change **rarely**.
    - **Output file format** must **never change**.

!!! tip "Review tone"
    Be explicit, patient, and educational. Assume the author is still learning; explain *what to change* and *why* with short examples or links.

!!! note "Long-term stewardship"
    Publication deadlines or short-term milestones must never override the mandate to protect Quokka's scientific integrity and technical direction. When in doubt, slow down, request revisions, or defer merges until the change meets the bar.

For quick definitions of terms used throughout this guide, see the [Glossary](glossary.md).

---

## Intake & Triage

1. **Require a complete PR description (plain language).**
    - Must include:
        - **Problem:** what bug/feature/refactor is addressed.
        - **Approach:** algorithms; **units** (e.g., cgs); **error tolerances** (e.g., relative error ≤ 1e-8); **boundary conditions** (periodic, inflow, outflow, reflecting; Dirichlet, Neumann, or periodic for elliptic solvers); **staggering** (cell-centered vs. face-centered).
        - **Stability impact:** does it touch `ParmParse` options, the output file format, or the problem-file interface? (Explain how.)
        - **Performance results (if applicable):** before/after timing or brief profiling on a test problem.
        - **Reproducer steps (if applicable):** exact build options (CMake), `ParmParse` inputs, MPI process count, and GPU backend (CUDA or HIP).

2. **Target the right branch and scope.**
   PRs should come from a fork/feature branch and target `development`. Looking at the last 100 squash merges and filtering out trivial updates (<50 changed lines, e.g., submodule bumps or workflow tweaks), a typical PR lands around **~238 changed lines across four files** (75th percentile ≈357 lines, 90th percentile ≈1.1k). Use that baseline: if a PR mixes unrelated changes, sprawls beyond ~8–10 files, or pushes well past ~400–500 lines of real code delta, ask for smaller, topic-focused PRs (“one PR = one story”).

---

## Correctness & Numerics

3. **Tests are mandatory.**
    - Provide:
        - **Unit/algorithm test:** checks a single function or kernel. Currently, there are no unit tests in Quokka.
        - **Physics test:** a small simulation test problem that runs in a few minutes. All tests in Quokka are currently physics tests.
        - **CI coverage:** our GitHub Actions matrix drives `canary run` against curated `ctest` lists on Linux/gcc (Release + ASAN), macOS/AppleClang, and ARM64 (Debug, Release, HWASan) plus two MHD-focused suites. Wire new problems into those lists (or `regression/quokka-tests.ini` for longer inputs) so they execute automatically. The GitHub Actions `hip` job is compile-only; runtime HIP coverage comes from `/azp run` on the Azure Pipelines self-hosted AMD GPUs. The Azure Pipelines CI also tests the CUDA backend on an NVIDIA GPU. Note any additional backend-specific manual validation you perform.

4. **Document numerical expectations in code/comments.**
   State valid input ranges, units, absolute/relative tolerances, conditioning notes (sensitivity to small input changes), **staggering** (cell-centered vs. face-centered), and the required number of **ghost cells** for correctness.

5. **Boundary conditions and conservation (if applicable).**
   Describe how boundary conditions are applied and show that conserved quantities (mass, momentum, energy) remain correct within tolerance (norms or integrals over the mesh data).

---

## Stability Policy

6. **Runtime `ParmParse` options (change rarely).**
    - **Default stance:** options are **stable**; avoid renaming/removing.
    - **If change is necessary:**
        - Provide **automatic compatibility** when feasible (accept old name, map to new, and warn).
        - Emit **clear deprecation warnings** that name the replacement and include a removal date/version.
        - Document the change prominently (release notes; “Upgrading” section).
        - Use the ADR process (below) for any non-trivial change in behavior or naming.

7. **Output file format (must never change).**
    - Do **not** modify existing structure, metadata, or semantics.
    - If new data must be written, use **additive** outputs that do not alter or invalidate existing files (e.g., auxiliary files or parallel sidecar outputs) so existing analysis tools remain valid.
    - Any proposal that risks incompatibility should be rejected outright or re-scoped to preserve the current format.

8. **Core hydro integrator (`QuokkaSimulation::advanceHydroAtLevel`).**

    !!! danger "Critical change protocol"
        - Treat every edit as **critical**: require an ADR with numerical justification, targeted regression coverage spanning CPU and at least one GPU backend, and explicit before/after plots or diagnostics for representative hydro problems.
        - Demand quantified performance data (solver wall time, per-stage timing, halo exchange cost) and call out any tolerable slowdowns or accuracy gains.
        - Double-check plotfile contents and conserved quantities for regressions before approving.

9. **Problem-file interface (may change occasionally).**
    - Changes are acceptable to support new physics/modules, but must be **intentional and documented**.
    - Require an **ADR** describing the motivation, alternatives, and migration plan.
    - Migrate all in-tree problem files in the same PR; out-of-tree problems are not supported and must supply their own migrations.

---

## Quokka-Specific Pitfalls to Avoid

!!! warning "Watch for these regressions-in-waiting"
    - **Renaming widely-used variables.** Mechanical symbol churn often leaves call sites or device code untouched, introducing subtle bugs in solver logic and diagnostics.
    - **Adding redundant physics implementations.** Multiple solvers for the same physical process explode our validation matrix; prefer extending or generalising existing kernels.
    - **Copy/pasting shared routines.** Duplicated flux or source-term code drifts out of sync; factor shared helpers instead so fixes land once.

---

## Reproducibility

10. **Define the reproducibility level and test it.**
    - **Bitwise identical:** outputs match exactly.
    - **Within a tolerance:** values may differ slightly but within bounds.
    - **Statistically equivalent:** stochastic runs match in distribution with fixed seeds.
        Control randomness with documented seeds; avoid rank-dependent draw order.

11. **Record run metadata (log/output).**
    Always include: git commit hash; AMReX version; compiler and **GPU backend** (CUDA or HIP) + versions; MPI version; GPU model; key build flags; full `ParmParse` dump; and domain decomposition. Keep minimal input decks with tests so others can rerun them.
    Plotfiles and checkpoints automatically persist provenance to `metadata.yaml`, which records `quokka_version`, `git_hash_quokka`, `git_hash_amrex`, and the current unit and constant blocks by default. Update or extend that node (via `simulationMetadata_`) when additional diagnostics are essential for reruns.

---

## CI Gates

12. **Automated checks must stay green (GitHub Actions).**
    - `CMake`: Ubuntu + GCC 11 Release with `-DENABLE_ASAN=ON`; builds the full 3D configuration and runs the main `canary` suite.
    - `CMake (macOS)`: macOS 14 + Apple Clang Release; runs the same `canary` list.
    - `linux-arm64-{debug,release,HWASan}`: ARM runners using Clang; run fast 1D smoke tests in Debug, Release, and HWASan configurations.
    - `warnings`: Ubuntu build with `-DWARNINGS_AS_ERRORS=ON`; fails on any new compiler warning.
    - `MHD-WaveTests` and `MHD-WaveTests-ASAN`: GCC Release and ASAN builds that rebuild only MHD-labelled targets and execute the MHD-focused `canary` subsets.
    - `hip`: ROCm/clang toolchain compile of the full codebase (GitHub Actions build-only sanity check).
    - `OpenPMD`: builds with `QUOKKA_OPENPMD=ON` and runs the 3D blast problem under MPI to exercise ADIOS2/openPMD output.
    - Static analysis & lint: `clang-tidy-review` (with follow-up comment job) and `codespell`.
    - Docs & metadata: `docs` (MkDocs build), `docs-toctree`, `dependency-review`, `codeql`, and `scorecard`.
    - **Azure Pipelines (self-hosted GPUs):** manual `/azp run` trigger by a maintainer launches CUDA and HIP runtime tests on secured NVIDIA/AMD runners; required for accelerator-sensitive changes.
    - **Nightly regression suite:** runs the `regression/` harness against reference plotfiles on a separate pipeline; failures mean the gold data and code outputs diverged and need investigation.
    - Most jobs go through `check_changes.yml`; docs-only or workflow-only PRs will auto-skip compute-heavy runs, so double-check locally if you bypass CI by rebasing after review.

    !!! tip "Docs-only changes"
        When automation skips builds, spot-check formatting and links locally (`./build_docs.sh`) before approving to avoid post-merge surprises.

---

## Architecture Decision Records

Use [Architecture Decision Records](adrs.md) for high-impact or hard-to-reverse changes—especially anything touching `ParmParse` options, the problem-file interface, or output formats. They capture competing options, supporting evidence, and the rollout plan so we can resolve debates quickly and revisit choices with context.

## Merge Policy

!!! warning "Need to undo a merge?"
    See the dedicated [PR Revert Policy](pr_revert_policy.md) for decision criteria and the step-by-step workflow.

13. **Be explicit, kind, and disciplined.**
    - Mark comments as **blocking** (must fix before merge) or **non-blocking** (suggestions).
    - If discussion becomes subjective (naming/style), time-box and move to an ADR.
    - **Timelines:**
        - Acknowledge a new PR within **5 business days**.
        - Provide a first substantive review within **14 days**.
        - For high-risk or broadly user-facing changes (`ParmParse`, output format, problem-file interface), require **two maintainer approvals**; otherwise **one** is sufficient.
        - Use **squash-and-merge** by default (keeps history clean).

---
