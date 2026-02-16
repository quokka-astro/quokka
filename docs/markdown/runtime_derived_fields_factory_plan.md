# Runtime Derived Fields Factory Plan

## Status
- Owner: TBD
- Last updated: 2026-02-16
- Stage: Planning

## Problem Statement
The current derived-field path is primarily compile-time via problem-specific specializations of `ComputeDerivedVar(...)`. This makes reusable analysis fields difficult to share across problems and forces rebuilds for configuration changes.

Goal: add a runtime factory-based derived field system so fields can be defined/configured from inputs and consumed by diagnostics/plotfile logic without each problem implementing `ComputeDerivedVar(...)`.

## High-Level Goals
- Introduce a factory-registered runtime interface for derived field providers.
- Keep diagnostics as pure consumers of fields (no provider-side output responsibility).
- Preserve backward compatibility with existing compile-time `ComputeDerivedVar(...)` behavior.
- Support particle deposition derived fields as the first concrete provider.

## Non-Goals (Phase 1)
- Replacing all compile-time derived fields immediately.
- Supporting face-centered runtime derived fields.
- Implementing arbitrary cross-level reduction fields.

## Requirements
- Runtime input configuration analogous to diagnostics.
- Deterministic and cycle-safe dependency resolution.
- Compatibility with AMR regrid/update lifecycle.
- No naming collisions with existing cell-centered, face-centered, or derived variables.
- GPU-safe implementation patterns (no unsafe host pointer captures).

## Proposed Architecture

### 1. Base Interface + Factory
Create `DerivedFieldBase` (factory-registered, similar to `DiagBase`) with:
- `init(prefix, fieldName)`
- `prepare(nlevels, geoms, grids, dmap, availableVars)`
- `addOutputs(std::vector<DerivedOutputSpec>&)`
- `addDependencies(std::vector<std::string>&)`
- `compute<problem_t>(lev, time, dstMF, dstComp, context)`

`DerivedOutputSpec` should include:
- output name
- component count
- centering (start with cell-centered only)
- required ghost cells

### 2. Runtime Manager in `AMRSimulation`
Add `RuntimeDerivedFieldManager` owned by `AMRSimulation`:
- parse `quokka.derived_fields = <list>`
- instantiate providers via factory (`type = ...`)
- gather output specs and dependencies
- validate names and build dependency graph
- topologically sort providers/outputs
- allocate/cache computed MultiFabs per level for current step/time

### 3. Dependency + Evaluation Model
- Construct DAG over runtime-derived outputs.
- Abort on cycles or missing dependencies.
- Evaluate lazily on demand with memoization for `(step,time,lev,field)`.
- Invalidate cache on new step or grid topology change.

### 4. Consumer Integration
Integrate runtime-derived resolution into both:
- diagnostics assembly path (`doDiagnostics()` and `diagMFVec` requests)
- plotfile variable assembly path (`PlotFileMFAtLevel_cc`)

Resolution order for a requested variable:
1. native cell-centered field
2. face-centered (averaged-to-cc) field
3. runtime-derived field manager
4. legacy compile-time `ComputeDerivedVar(...)`

### 5. Context Object for Providers
Provide a narrow `DerivedFieldComputeContext` API for providers:
- geometry per level
- const access to state MultiFabs
- lookup for already-computed runtime-derived dependencies
- optional particle access services via simulation pointer wrappers

Avoid exposing large mutable simulation internals directly.

## Particle Deposition Provider (First Use Case)
Implement `DerivedParticleDeposition` provider:
- Input config:
  - `particle_types = CIC StochasticStellarPop ...`
  - `deposit_fields = mass`
  - optional naming prefix
- Outputs:
  - e.g. `particle.CIC.mass_density`
- Compute behavior:
  - zero destination components
  - call deposition helpers in `src/particles/particle_deposition_utils.hpp`
  - deposit into output MultiFab components

This provider must not write files or emit standalone diagnostic output.

## Input Configuration Sketch

```ini
quokka.derived_fields = partdep

quokka.partdep.type = DerivedParticleDeposition
quokka.partdep.particle_types = CIC
quokka.partdep.deposit_fields = mass
quokka.partdep.prefix = particle

quokka.diagnostics = hist
quokka.hist.type = DiagPDF
quokka.hist.int = 10
quokka.hist.var_names = particle.CIC.mass_density
```

## Implementation Plan

### Phase 0: Design + Scaffolding
- Add base class and factory registration.
- Add manager class with parsing and provider lifecycle hooks.
- Add output/dependency metadata structures.
- Add docs for input schema and naming rules.

Exit criteria:
- code compiles with no providers enabled.
- manager can parse empty/non-empty config.

### Phase 1: Evaluation Engine + Integration
- Implement DAG build, validation, and topo sort.
- Implement per-step/per-level cache and invalidation.
- Integrate lookup into diagnostics and plotfile assembly paths.
- Keep legacy `ComputeDerivedVar(...)` fallback path intact.

Exit criteria:
- runtime-derived fields can be requested by diagnostics.
- startup aborts clearly on cycles/missing deps/collisions.

### Phase 2: Particle Deposition Provider
- Implement `DerivedParticleDeposition` provider.
- Map particle type names to containers safely.
- Support `mass` deposition first.

Exit criteria:
- particle deposition field appears in diag/plot output when requested.
- no direct provider output files generated.

### Phase 3: Hardening + Migration
- Add tests for dependency ordering and error paths.
- Add regression case where DiagPDF consumes particle deposition runtime field.
- Document migration guidance from `ComputeDerivedVar(...)` specializations.

Exit criteria:
- existing tests pass.
- new tests cover runtime provider path.

## Validation and Testing Plan
- Unit/functional checks:
  - provider discovery and factory creation
  - collision detection
  - cycle detection
  - missing dependency detection
- Integration checks:
  - runtime-derived field requested by `DiagPDF`
  - runtime-derived field requested by `DiagPlotfile`
  - mixed mode: runtime provider + legacy compile-time derived var
- Performance sanity:
  - ensure field computed once per step/time/level when cached

## Risks and Mitigations
- Risk: variable namespace collisions.
  - Mitigation: strict validation at init with actionable errors.
- Risk: stale cache across regrid/time changes.
  - Mitigation: explicit invalidation hooks in `updateDiagnostics()` and step transitions.
- Risk: over-coupling providers to simulation internals.
  - Mitigation: narrow context interface and helper services.
- Risk: AMR semantics bugs (coarse/fine consistency).
  - Mitigation: use existing averaging/fill-boundary patterns used for derived vars.

## Open Decisions
- Should runtime providers be allowed to emit multi-component vector outputs in phase 1?
- Should provider names map one-to-one to output field names, or many outputs per provider?
- Should runtime-derived fields be allowed to depend on legacy compile-time derived vars bidirectionally?
- How strict should naming conventions be (`particle.CIC.mass_density` vs `CIC_mass`)?

## Task Checklist
- [ ] Add `DerivedFieldBase` and registration macros.
- [ ] Add manager class and config parsing.
- [ ] Add output/dependency metadata + validation.
- [ ] Integrate runtime lookup in diagnostics assembly.
- [ ] Integrate runtime lookup in plotfile assembly.
- [ ] Implement cache + invalidation.
- [ ] Implement `DerivedParticleDeposition` provider.
- [ ] Add regression test input and expected behavior.
- [ ] Add user docs for config and naming.

## Suggested File Touch Points (initial)
- `src/simulation.hpp`
- `src/io/` (new derived-field base + manager)
- `src/particles/particle_deposition_utils.hpp` (reuse)
- `src/CMakeLists.txt` (register new files)
- `docs/markdown/parameters.md` (new runtime derived config docs)

