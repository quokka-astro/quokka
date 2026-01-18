# Plan: PureSoA ParticleContainers + PolymorphicArenaAllocator

## Goals
- Move Quokka physics particle containers to AMReX PureSoA (positions + attributes all SoA).
- Use `amrex::PolymorphicArenaAllocator` to select arenas at runtime and avoid hard-coding device/managed/pinned.
- Preserve existing physics behavior, I/O semantics, and diagnostics.

## Scope
- In-scope: Quokka physics particle containers and helpers in `src/particles/`, `src/simulation.hpp`, and particle-related problem tests.
- Out-of-scope (unless requested): `amrex::AmrTracerParticleContainer` (no PureSoA alias in AMReX; keep AoS or handle separately).

## References (AMReX)
- PureSoA docs: first `AMREX_SPACEDIM` real components are positions; use `ParticleTileData` and `operator[]` for layout-agnostic access.
- `SOAParticle` test: `ParticleContainerPureSoA`, `ParticleTileData` access, `make_alike<PinnedArenaAllocator>()`, `ParticleReduce` with PTD.
- `NamedSoAComponents` test: `SetSoACompileTimeNames`, `AddRealComp`, `AddIntComp`, and name-based access.
- `RedistributeSOA` test: SoA initialization using host vectors, `ParticleIDWrapper`, `ParticleCPUWrapper`, runtime comps.
- `CheckpointRestartSOA` and `InitRandom`: `InitRandom` + `Checkpoint/Restart` with PTD access patterns.

## Proposed Data Layout
- Keep logical component indices (e.g., mass index = 0) as "extra" indices.
- For PureSoA, actual SoA index = `logical_index + AMREX_SPACEDIM`.
- Add a small helper (new header or in `particle_types.hpp`):
  - `realOffset<ContainerType>()` -> `AMREX_SPACEDIM` for SoA, `0` for AoS.
  - `realIndex<ContainerType>(logical)` -> `logical + realOffset`.
  - `numExtraReal(container)` -> `container->NumRealComps() - realOffset`.
  - `numExtraInt(container)` -> `container->NumIntComps()` (same for AoS/SoA).
- This keeps I/O (positions first, then extra data) and existing indices consistent.

## Container and Allocator Changes
- Replace `amrex::AmrParticleContainer<...>` with a PureSoA alias, e.g.:
  - `using AmrParticleContainerPureSoA = amrex::AmrParticleContainer_impl<amrex::SoAParticle<NArrayReal,NArrayInt>, NArrayReal, NArrayInt, amrex::PolymorphicArenaAllocator>;`
- For each particle type, set `NArrayReal = AMREX_SPACEDIM + extra_real_comps` and `NArrayInt = extra_int_comps`.
- Update iterator aliases to use `ContainerType::ParIterType` (or `amrex::ParIterSoA`).
- Add `particles.arena` ParmParse option (e.g., `default|device|managed|pinned|host`) and apply via `SetArena()` before any particle tile is defined.
  - Ensure `SetArena()` is called in `AMRSimulation::InitPhyParticles` and restart path before `Define`, `InitRandom`, or `InitFromAsciiFile`.

## Code Refactor Plan

### Phase 1: Layout Helpers and Type Aliases
- Add helper functions for SoA offsets and component counts (new header or `particle_types.hpp`).
- Update particle container typedefs in `src/particles/particle_types.hpp` to PureSoA + PolymorphicArenaAllocator.
- Update any direct use of `ParticleType::NReal`/`NInt` to the helper functions or `container->NumRealComps()` / `NumIntComps()`.

### Phase 2: Core Particle Algorithms (SoA Access)
Replace all AoS accesses (`GetArrayOfStructs`, `m_aos`, raw AoS pointers) with PureSoA patterns:
- `src/particles/PhysicsParticles.hpp`
  - `computeStellarMass`, `computeMaxParticleSpeed`, `driftParticles`, `kickParticles`, `splitParticles`, `tagCellsAroundParticles`.
  - Use `ParticleTileData` and `realIndex()` for rdata access.
- `src/particles/particle_deposition.hpp`
  - SN deposition and evolution-stage updates (currently AoS).
  - Use `ptd = pti.GetParticleTile().getParticleTileData()` and `SoAParticle` wrappers or direct `rdata` arrays.
- `src/particles/particle_creation.hpp`
  - Redesign `ParticleCreator` interface to fill SoA arrays (pass `ParticleTileData` + base offset).
  - Replace AoS `particles().data()` with SoA writes (`ptile.pos`, `ptd.rdata`, `ptd.idata`, `ptd.idcpu`).
- `src/particles/particle_destruction.hpp`
  - Mark invalid via `ptd.idcpu` or `SoAParticle` wrapper.
- `src/particles/particle_accretion.hpp`
  - Replace AoS pointer loops with `ptd` access; apply `realIndex()` offset for mass/velocity components.
- `src/particles/particle_radiation.hpp` and `particle_update.hpp`
  - Update component index usage to `realIndex()` and `numExtraReal()` checks.

### Phase 3: I/O and Diagnostics
- `src/particles/particle_IO.hpp`:
  - Replace AoS copies with PureSoA-friendly flows:
    - Use `make_alike<PinnedArenaAllocator>()`, `copyParticles()`, and `Redistribute()` for rank-0 gather.
    - Extract positions and extra real data from PTD: positions from `ptd.rdata(0..AMREX_SPACEDIM-1)`, extra from `ptd.rdata(AMREX_SPACEDIM + i)`.
  - Preserve output format (positions first, then extra data).
  - Use `ContainerType::ParticleType::is_soa_particle` to skip positions when iterating over `rdata`.
- Set compile-time SoA names for plotfile readability:
  - Use `SetSoACompileTimeNames({x,y,z, mass, vx, ...}, {int_names...})` after container construction.

### Phase 4: Problem Tests and Initialization
- Update AoS usage in `src/problems/*` tests:
  - Replace `GetArrayOfStructs` and direct AoS access with `ParticleTileData` + `SoAParticle`.
  - Apply `realIndex()` offsets for mass/velocity/luminosity indices.
- Confirm `InitFromAsciiFile` and `InitRandom` calls use extra component counts (positions handled internally by AMReX for PureSoA).

### Phase 5: Validation and Regression
- Build and run targeted tests:
  - Particle-focused tests: `ParticleCreation`, `ParticleAccretion`, `ParticleRadiation`, `ParticleSink`, `GravRadParticle3D`, `BinaryOrbitCIC`, `SphericalCollapse`.
  - Regression suite updates in `regression/quokka-tests.ini` if outputs change (plotfile component ordering).
- Add a small SoA-specific unit test (optional) mirroring AMReX `SOAParticle` patterns for Quokka (PureSoA + PolymorphicArenaAllocator).

## Risks and Mitigations
- Indexing errors due to SoA position offset:
  - Mitigate with a single `realIndex()` helper and consistent usage.
- I/O ordering changes:
  - Keep output arrays as `[pos..., extra...]` and update diagnostics accordingly.
- Arena misconfiguration (Polymorphic allocator requires `SetArena()` before tile definition):
  - Enforce `SetArena()` on container creation paths and document new input option.

## Open Questions
- Should tracer particles also migrate (requires custom PureSoA alias for `AmrTracerParticleContainer` or leave AoS)?
- Preferred default arena (`The_Arena()` vs `The_Device_Arena()` vs `The_Managed_Arena()`), and how to expose it in inputs.
