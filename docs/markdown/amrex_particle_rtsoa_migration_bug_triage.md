# AMReX particle bugs affecting the RTSoA migration

This document triages known AMReX particle bugs that can affect Quokka's migration from compile-time array-of-structs particle containers to `amrex::ParticleContainerRTSoA<>`.

The source inventory is the [`particle-function-audit` issue collection](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/README.md). The classifications in that inventory describe current Quokka, whose particle payloads are stored in AoS fields. Several bugs classified there as not affecting Quokka become applicable as soon as Quokka adopts a pure runtime struct-of-arrays layout.

This triage was reviewed against audit commit `7fd547258378cabcf41f0a33078db2cc0628e34a` and Quokka's vendored AMReX commit `3c3bbaa34cd3423fe96e6cc601a33b42bfd5238d`. Recheck both the issue inventory and the vendored source after either revision changes.

## Triage policy

The priorities below describe migration risk, not the general severity assigned by the AMReX audit.

- **P0 -- migration blocker:** fix in AMReX before converting production containers. A direct migration would otherwise fail to compile, corrupt particle state, lose particles, or produce invalid restart data.
- **P1 -- required for feature parity:** fix before declaring the migration complete. Quokka can temporarily avoid the affected optional path or enforce a narrower contract.
- **P2 -- hardening:** fix upstream, but a Quokka schema wrapper can prevent the invalid call in the meantime.
- **Excluded:** confirmed bugs that do not intersect the proposed RTSoA call graph. They are not part of the migration gate.

An AMReX fix is considered complete only after its regression test instantiates a pure runtime-SoA container. Hybrid or AoS-only coverage is insufficient.

## P0: migration blockers

### P0.1 Define runtime layouts before populating tiles

**Bug:** [Particle initializers bypass runtime tile definition](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/likely_not_affected/particle-initializers-bypass-tile-definition.md)

Several initializers insert through `GetParticles()[key]` or `m_particles[key]`. This default-constructs a tile without the container's runtime component counts, names, or arena. Resizing or copying data into that tile is invalid for `ParticleContainerRTSoA<>`.

**Required AMReX change:** every initializer must obtain destination storage through `DefineAndReturnParticleTile`, or an equivalent helper that applies the complete container schema and allocator before resizing.

**Acceptance test:** initialize a container with at least `AMREX_SPACEDIM + 2` runtime real components and two runtime integer components, then verify the count, names, values, arena, and redistribution result on every populated tile.

### P0.2 Make the generic initializers available for pure RTSoA

**Bug:** [Several generic initializers cannot instantiate for pure-SoA containers](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/legacy-initializers-pure-soa-unavailable.md)

Quokka currently depends on `InitFromAsciiFile`, `InitRandom`, and `InitOnePerCell`. The migration cannot preserve those call sites while these routines instantiate AoS-only particle objects or otherwise assume an AoS payload.

**Required AMReX change:** implement pure-SoA paths for the initializers Quokka uses, with ID/CPU stored in the RTSoA identity array and positions stored in runtime real components `0..AMREX_SPACEDIM-1`. APIs that cannot support RTSoA must reject it with an explicit compile-time constraint rather than failing inside their implementation.

**Acceptance test:** compile and run pure-RTSoA tests for ASCII, random, and one-per-cell initialization on CPU and GPU builds.

### P0.3 Correct pure-SoA random initialization

**Bugs:**

- [Pure-SoA random initialization skips integer components zero and one](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/random-init-pure-soa-int-components.md)
- [Parallel pure-SoA random initialization consumes two IDs per particle](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/random-init-pure-soa-consumes-extra-id.md)

The first bug silently leaves two user integer fields unset. The second makes the generated ID sequence depend on a redundant temporary-ID allocation. Both become directly reachable in Quokka's spherical-collapse initialization after conversion.

**Required AMReX change:** populate every declared runtime integer component starting at index zero and allocate exactly one ID for each accepted particle.

**Acceptance test:** initialize a known count on multiple MPI ranks; verify every integer value, uniqueness, exact ID progression, and parity between serialized and parallel generation.

### P0.4 Initialize every particle component deterministically

**Bug:** [File and random initializers leave unspecified components uninitialized](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_affected/particle-initializers-uninitialized-components.md)

This already affects AoS Quokka containers and becomes more dangerous with a shared runtime schema, where omitted fields remain allocated and can appear valid to later kernels and output routines.

**Required AMReX change:** define a deterministic policy for fields not supplied by the input source. Prefer explicit defaults from `ParticleInitData`; otherwise initialize to zero. Do not leave device or host allocation contents as particle state.

**Acceptance test:** initialize a schema with more fields than the input provides and verify every omitted field bit-for-bit on CPU and GPU.

### P0.5 Count pure-SoA particles correctly in pre/post output

**Bug:** [CheckpointPre reads device AoS storage and counts pure-SoA containers as empty](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/likely_affected/checkpoint-pre-device-and-pure-soa-count.md)

`CheckpointPre` traverses `GetArrayOfStructs()`. A pure RTSoA container therefore reports zero particles even when populated, producing inconsistent metadata when `particles.use_prepost=1`.

**Required AMReX change:** compute the valid-particle count and next-ID metadata through layout-independent tile data or existing layout-independent counting functions. The implementation must not dereference device storage from a host loop.

**Acceptance test:** perform a pre/post checkpoint and restart of a nonempty pure-RTSoA container on CPU and GPU, including invalid particles and more than one AMR level.

### P0.6 Apply instance I/O options to every container

**Bug:** [Particle I/O options are applied only to the first container instance](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/likely_not_affected/particle-container-instance-io-options.md)

`ParticleContainer_impl::Initialize()` uses a function-static `initialized` guard while assigning `usePrePost` and `doUnlink`, which are instance members. Once all Quokka particle populations use the same `ParticleContainerRTSoA<>` specialization, only the first constructed population receives the parsed values.

**Required AMReX change:** separate specialization-wide configuration from instance initialization. Parse or cache global defaults once, then copy all instance-owned options into every newly constructed container.

**Acceptance test:** construct at least two containers of the same RTSoA specialization and verify identical `GetUsePrePost()` and `GetUseUnlink()` values for both settings enabled and disabled.

## P1: required for feature parity

### P1.1 Make ASCII output layout-independent

**Bug:** [ASCII output writes pure-SoA containers as empty](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/ascii-output-pure-soa-empty.md)

AMReX `WriteAsciiFile` counts and serializes only AoS records. Quokka currently has its own particle text-output helper, which must also be converted, so AMReX ASCII output is not required for the first migrated container. It should nevertheless work before pure RTSoA is treated as feature-complete.

**Required AMReX change:** serialize through layout-independent tile data and write positions, selected runtime real fields, runtime integer fields, and ID/CPU according to a documented format.

**Temporary containment:** do not call `WriteAsciiFile`; retain a Quokka-owned, schema-aware text writer.

### P1.2 Validate checkpoint names and component masks before indexing

**Bugs:**

- [Checkpoint indexes caller component names before validating their lengths](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/likely_not_affected/checkpoint-component-name-size.md)
- [Particle I/O does not validate component-mask lengths](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/likely_not_affected/particle-io-component-mask-size.md)

Runtime schemas make name and mask lengths dynamic. A mismatch can read beyond a vector in optimized builds or produce a header whose component count disagrees with the particle records.

**Required AMReX change:** add always-on, exact-length validation at the public I/O boundary before indexing a name or mask, creating directories, or mutating pre/post state. Pure RTSoA validation must account for position components separately from user real components.

**Temporary containment:** generate names and masks exclusively from one immutable Quokka `ParticleSchema` and assert exact lengths before calling AMReX.

### P1.3 Reject selective checkpoints that cannot be restarted

**Bug:** [Selective checkpoints cannot be restarted by the same container type](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/likely_not_affected/selective-checkpoint-cannot-restart.md)

The checkpoint API permits component masks, but `Restart` requires the file to contain the container's complete schema.

**Required AMReX change:** either reject deselection in `Checkpoint` with an always-on diagnostic or extend the format and reader with component identities and explicit defaults. Plotfile output may continue to support selection.

**Temporary containment:** Quokka checkpoints must always write every schema component.

### P1.4 Define restart behavior for nonempty containers

**Bug:** [Restart silently appends to an already populated container](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/likely_not_affected/restart-appends-existing-particles.md)

Appending is especially hazardous when several populations have the same C++ container type and are managed through a common wrapper.

**Required AMReX change:** make the contract explicit. Prefer an always-on empty-container precondition, or provide a deliberately named append mode.

**Temporary containment:** the Quokka wrapper must assert that the destination contains no particles before calling `Restart`.

### P1.5 Prevent AoS-only transfers from accepting SoA schemas

**Bug:** [AoS transfer overloads are unsafe for containers with SoA components](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/particle-aos-transfer-soa-components.md)

The affected `AddParticlesAtLevel`, `CreateVirtualParticles`, and `CreateGhostParticles` overloads cannot represent runtime fields. Reusing an AoS temporary during Quokka's restart-refinement or particle-splitting conversion would lose data or access empty arrays.

**Required AMReX change:** constrain AoS overloads to schemas with no SoA components and direct component-bearing callers to complete `ParticleTileType` overloads.

**Temporary containment:** prohibit AoS bridge objects in migrated Quokka code and copy complete tiles or tile-data records instead.

## P2: upstream hardening with Quokka containment

### P2.1 Correct RTSoA component-index validation

**Bug:** [Runtime-only particle tile accepts negative and non-position indices](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/particle-tile-rt-index-bounds.md)

Negative indices pass upper-bound-only checks, and `pos(dir)` validates against the total real-component count rather than `AMREX_SPACEDIM`. A bad direction can therefore alias the first physics field.

**Required AMReX change:** enforce both lower and upper bounds and restrict position access to spatial dimensions.

**Temporary containment:** expose typed Quokka schema keys and fixed-dimension position accessors rather than public signed component indices.

### P2.2 Reject negative RTSoA sizes and component counts

**Bugs:**

- [Runtime-only particle tiles accept negative sizes and component counts](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/particle-tile-rt-negative-sizes.md)
- [Runtime component resize accepts negative counts](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/runtime-component-negative-size.md)

**Required AMReX change:** validate signed sizes before mutating state or converting them to allocation sizes.

**Temporary containment:** validate the Quokka schema once and use nonnegative size types internally.

### P2.3 Keep runtime names synchronized with storage

**Bug:** [Runtime component resize leaves the name vectors out of sync](https://github.com/BenWibking/amrex/blob/particle-function-audit/issues/def_not_affected/runtime-component-resize-name-desync.md)

**Required AMReX change:** update names atomically with storage and communication masks, requiring explicit names for newly added fields.

**Temporary containment:** freeze each Quokka particle schema after container construction. Do not call `ResizeRuntimeRealComp` or `ResizeRuntimeIntComp`.

## Additional migration hazard not represented in the audit

`RTSoAParticle` uses the shared `NextIDRTSoA` counter for every `ParticleContainerRTSoA<>` instance. This needs an explicit compatibility test when restarting old Quokka checkpoints, whose AoS particle specializations may have recorded different next-ID values. Sequential calls to `Restart` must not leave the shared counter below the maximum required by any restored population.

Until AMReX defines a multi-container restart contract, Quokka should collect the next-ID requirement across every restored particle dataset, perform a communicator-wide maximum reduction, and set the shared RTSoA counter once after all populations have restarted.

## Excluded from the migration gate

The following confirmed bug families do not need to be fixed for the initial migration because Quokka does not use their affected paths:

- asynchronous particle output, which Quokka explicitly rejects;
- neighbor-particle containers and neighbor lists;
- virtual and ghost particle creation;
- `make_alike` with a custom `CellAssignor`;
- direct `filterParticles` and `filterAndTransformParticles` calls;
- particle binning and sorting utilities;
- split-`ParallelContext` particle initialization and I/O.

These exclusions must be revisited if the RTSoA wrapper begins exposing any corresponding API. Async output in particular must remain disabled until its separate audit issues are fixed and covered by round-trip tests.

## Migration acceptance suite

The AMReX fixes above should be tested together with a representative pure-RTSoA schema:

- positions in runtime real components `0..AMREX_SPACEDIM-1`;
- at least two additional real components;
- at least two integer components;
- two distinct container instances using the same RTSoA specialization;
- empty and nonempty tiles, invalid particles, and multiple AMR levels;
- one-rank and multi-rank execution;
- CPU and GPU builds.

The end-to-end test should cover:

1. ASCII, random, and one-per-cell initialization.
2. Redistribution and component preservation.
3. Normal plotfile and checkpoint output.
4. Pre/post checkpoint output.
5. Restart into an empty container.
6. Exact schema names, masks, values, particle counts, and IDs after restart.
7. Multiple particle populations followed by creation of a new particle, proving that the restored shared next-ID state is safe.

The Quokka migration should not be declared complete until this matrix passes against the exact AMReX revision vendored by Quokka.
