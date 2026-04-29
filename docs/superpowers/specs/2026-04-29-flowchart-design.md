# Flowchart Redesign — Design Spec

**Date:** 2026-04-29
**Branch:** chong/doc/new-flow-chart

## Goal

Rewrite `docs/markdown/flowchart.md` with a single, comprehensive PlantUML activity diagram that accurately and completely reflects the Quokka simulation call chain. The current Mermaid diagram has broken syntax (duplicate subgraph names), disconnected nodes, and is missing large portions of the actual code flow.

## Audience

Both new developers onboarding to the codebase and experienced contributors using it as a quick algorithmic reference.

## Tool Choice

**PlantUML only** for the flowchart diagram (replacing Mermaid). PlantUML is supported by the mdBook `mdbook-plantuml` preprocessor and provides native constructs for loops (`repeat...repeat while`), conditionals (`if/else/endif`), and grouping (`group...end group`) that express the actual control flow without arrow spaghetti.

Mermaid support in `mermaid-init.js` and `book.toml` stays untouched — other documentation pages use it.

## Diagram Structure

One PlantUML activity diagram covering the full call chain top-to-bottom:

### 1. Preamble
- `setInitialConditions()`
- Entry into `evolve()` main time loop

### 2. Inside `evolve()` — main time loop (`partition`)
1. `computeTimestep()`
2. `computeBeforeTimestep()` *(user hook)*
3. Particle leapfrog kick ×1 *(3D + particles only)*
4. `timeStepWithSubcycling(lev=0)` — see section 3
5. Particle drift
6. `ellipticSolveAllLevels()` *(self-gravity Poisson solve, if enabled)*
7. Particle leapfrog kick ×2 + `updateParticleProperties()` + `particleMeshInteraction()` + `destroyParticles()` *(3D only)*
8. `computeAfterTimestep()` *(user hook)*
9. Write plotfile / checkpoint *(if step/time interval reached)*

### 3. `timeStepWithSubcycling(lev)` (`partition`)
- **if** `regrid_int > 0` and step matches interval → `AMRCore::regrid()`
- `advanceSingleTimestepAtLevel(lev)` — see section 4
- **repeat** for `i = 1..nsubsteps[lev+1]`: recursive call `timeStepWithSubcycling(lev+1)` *(AMR subcycling)*
- `FluxRegister::Reflux()` *(flux conservation across coarse/fine interface)*
- `AverageDownTo(lev)` *(average fine level data down to coarse)*

### 4. `advanceSingleTimestepAtLevel(lev)` (`partition`)
1. Swap `state_old` ↔ `state_new`
2. `CheckHydroStates` *(before update)*
3. **if** hydro enabled → `advanceHydroAtLevelWithRetries()` — see section 5; **else** copy hydro vars old→new
4. `CheckHydroStates` *(after hydro)*
5. **if** radiation enabled → `subcycleRadiationAtLevel()` — see section 6
6. `CheckHydroStates` *(after radiation)*
7. `computeAfterLevelAdvance()` *(user hook)*
8. `CheckHydroStates` *(after user work)*

### 5. `advanceHydroAtLevelWithRetries()` + `advanceHydroAtLevel()` (`partition`)

**Retry loop** (`repeat...repeat while`):
- Call `advanceHydroAtLevel(dt)`
- On failure: halve `dt` and retry

**Inside `advanceHydroAtLevel()`:**
1. `addStrangSplitSourcesWithBuiltin(dt/2)`:
   - Cooling *(if enabled: resampled cooling table)*
   - Chemistry / nuclear burn *(if enabled)*
   - Turbulence driving *(if enabled and `t < t_stop`)*
   - Dust drag *(if dust enabled)*
   - `addStrangSplitSources()` *(user hook)*
2. `fillBoundaryConditions()`
3. **RK2-SSP Stage 1** — forward Euler flux update → `state_inter`
4. `fillBoundaryConditions()`
5. **RK2-SSP Stage 2** — corrector: ½(`state_old` + `state_inter` + dt·F(`state_inter`)) → `state_new`
6. `addStrangSplitSourcesWithBuiltin(dt/2)` *(same sub-steps as step 1)*

### 6. `subcycleRadiationAtLevel()` — IMEX PD-ARS (`partition`)
1. `computeNumberOfRadiationSubsteps()` → `nsubSteps`, `dt_rad`

**Substep loop** (`repeat...repeat while i < nsubSteps`):
- **if** `i > 0` → `swapRadiationState()` *(copy rad vars new→old)*
- *Stage 1: trivial — U⁽¹⁾ = Uⁿ (skipped)*
- **Stage 2 — explicit ForwardEuler + implicit coupling:**
  1. Copy `state_new` → `state_tmp1`
  2. `advanceRadiationForwardEuler(dt · Aex₂₁)` → `state_tmp1_rad`
  3. `SetRadEnergySource()` + particle radiation deposition *(3D)*
  4. `AddSourceTermsSingleGroup/MultiGroup(dt · Aim₂₂)` *(implicit Newton–Raphson: matter–radiation coupling)*
- **Stage 3 — explicit MidpointRK2 + implicit coupling:**
  1. `advanceRadiationMidpointRK2(dt)` *(uses `state_tmp1` as U⁽²⁾)*
  2. Shu-Osher gas combination: `state_new_gas` ← ½·`state_new` + ½·`state_tmp1`
  3. `SetRadEnergySource()` + particle radiation deposition *(3D)*
  4. `AddSourceTermsSingleGroup/MultiGroup(dt · Aim₃₃)` *(implicit Newton–Raphson)*

## Files Changed

| File | Change |
|------|--------|
| `docs/markdown/flowchart.md` | Full rewrite: single `plantuml` fenced code block |
| `scripts/bash/install_mdbook.sh` | Add `cargo install mdbook-plantuml --version 2.0.0` |
| `docs/book.toml` | Add `[preprocessor.plantuml]` section |
| `.gitignore` | Add `.superpowers/` |

Mermaid support (`docs/javascripts/mermaid-init.js`, `additional-js` in `book.toml`) is left untouched.

## PlantUML Constructs

| Need | Construct |
|------|-----------|
| Function grouping | `partition "FunctionName()" { ... }` *(activity diagram syntax)* |
| Retry / substep loops | `repeat ... repeat while (condition)` |
| Conditional physics | `if (enabled?) then (yes) ... else ... endif` |
| Inline annotations | `note right: ...` |
| IMEX stage labels | Named activity boxes |
