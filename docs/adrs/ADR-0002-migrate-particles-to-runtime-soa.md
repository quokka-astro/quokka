# ADR-0002: Migrate Quokka Particles to Runtime SoA
Date: 2026-07-22 • Status: Proposed

## Context

Quokka currently represents each particle population with a different instantiation of `amrex::AmrParticleContainer<NReal, NInt>`. Positions, identity, real-valued physics fields, and integer physics fields are accessed through AoS particle objects. The affected populations are:

- radiation particles;
- gravitating CIC particles;
- gravitating radiation particles;
- stochastic stellar-population particles;
- star particles;
- sink particles;
- test particles; and
- tracer particles, currently implemented with `amrex::AmrTracerParticleContainer`.

This design makes each combination of particle fields a separate C++ type. Radiation-group counts and stellar-model extensions create additional template instantiations. Particle algorithms are consequently coupled to compile-time component counts and AoS-specific interfaces such as:

- `GetArrayOfStructs()`;
- `ParticleType::NReal` and `ParticleType::NInt`;
- `ParticleType*` ranges;
- `p.rdata(i)` and `p.idata(i)` using compile-time payload indices; and
- pointers such as `&p.rdata(0)`, which assume that all fields belonging to one particle are contiguous.

The last assumption is fundamentally incompatible with SoA storage: adjacent addresses belong to adjacent particles in one component, not to successive components of one particle.

AMReX provides `amrex::ParticleContainerRTSoA<>`, whose positions and physics data are runtime-defined SoA components. It offers one concrete container type regardless of a population's schema and provides particle-like proxies through `ParticleTileDataRT::operator[]` for GPU kernels.

Adopting RTSoA would reduce particle-container template proliferation and make the storage model uniform, but it is not a type-alias substitution. The current Quokka particle implementation contains AoS assumptions in creation, drift/kick updates, accretion, destruction, deposition, radiation, stellar evolution, splitting, diagnostics, text output, checkpoint/restart helpers, and tracer advection.

The migration is also a prerequisite for future pyAMReX bindings for Quokka particles. The [AMReX/pyAMReX particle roadmap](https://github.com/AMReX-Codes/pyamrex/issues/460) identifies a runtime-component-only pure-SoA particle type as its third modernization step for 2026 and later. Its goal is for AMReX applications to use the same particle-container type, improving runtime compatibility and modularity while minimizing type complexity and compile/link time. The current Quokka matrix of population- and problem-specific container template instantiations does not satisfy that direction: every schema variation would require another concrete C++ binding, and Python could not discover fields uniformly at runtime. A single RTSoA container type plus explicit runtime schema metadata provides a stable binding boundary across particle populations, radiation-group counts, and stellar-model extensions.

### Storage-index mismatch

For `ParticleContainerRTSoA<>`, runtime real components `0..AMREX_SPACEDIM-1` store particle positions. Quokka's existing real-component enums use zero-based **logical physics indices**; for example, logical real component zero is commonly `mass`. Reinterpreting those enums as raw RTSoA indices would alias physics fields with positions.

Integer components do not have an analogous position prefix, but they still require schema validation and names.

### AMReX correctness constraints

The AMReX particle audit identified initialization, I/O, and runtime-layout defects that become applicable after the conversion. The migration gates and temporary containment policies are recorded in [AMReX particle bugs affecting the RTSoA migration](../markdown/amrex_particle_rtsoa_migration_bug_triage.md).

In particular:

- several generic initializers cannot instantiate for pure SoA or populate undefined runtime tiles;
- pure-SoA random initialization mishandles integer fields and IDs;
- `CheckpointPre` counts only AoS particles;
- per-instance I/O options are initialized only on the first instance of a specialization;
- AMReX ASCII output treats pure-SoA containers as empty; and
- all `ParticleContainerRTSoA<>` instances share `NextIDRTSoA` state.

The ADR assumes that the P0 AMReX issues are fixed in the vendored revision before production containers are converted. Quokka will still enforce its own schema and lifecycle invariants rather than exposing the affected low-level APIs directly.

### Compatibility constraints

The migration must preserve:

- particle directory and population names in plotfiles and checkpoints;
- particle field names, order, types, and units;
- restart compatibility with existing supported AoS checkpoints;
- numerical results within each test's existing tolerance;
- MPI, CPU, CUDA, and HIP support;
- existing runtime parameters and defaults; and
- the problem-generator semantics of each particle population; and
- a stable, runtime-introspectable particle interface suitable for future pyAMReX bindings.

No particle output-format change is authorized by this ADR.

### Goals

- Use one AMReX runtime-SoA container representation for every Quokka particle population.
- Make particle schemas explicit, named, immutable, and independently testable.
- Make hot particle kernels layout-independent and GPU-safe.
- Remove AoS-only data movement and per-particle contiguous-component assumptions.
- Preserve existing checkpoint, plotfile, units, and problem behavior.
- Permit particle algorithms to operate uniformly across populations when their schemas provide the required capabilities.
- Provide one concrete container type and schema-introspection model that future pyAMReX bindings can expose without binding every Quokka template specialization.

### Non-goals

- Redesign particle physics, deposition methods, accretion prescriptions, or stellar models.
- Change particle field precision.
- Change checkpoint or plotfile formats.
- Enable AMReX asynchronous output, neighbor particles, or ghost/virtual particle APIs.
- Make particle schemas mutable after construction.
- Generalize Quokka to support arbitrary user-defined schemas from runtime input files.
- Implement the pyAMReX bindings themselves; this ADR establishes the required C++ architecture and binding contract.

## Options

### Option A: Retain the compile-time AoS containers

Keep the existing aliases and refactor only isolated particle utilities.

This has the lowest immediate implementation risk but preserves template proliferation, AoS coupling, and duplicated container-specific code. It does not achieve the migration objective.

### Option B: Use compile-time pure-SoA containers

Replace each AoS type with `ParticleContainerPureSoA<NReal, NInt>` while retaining a distinct template instantiation for each schema.

This improves memory layout but keeps component counts in the type system, retains most template growth, and does not provide one uniform container type. Radiation-group and stellar-model variations would continue to create distinct container types, each requiring separate binding treatment in pyAMReX.

### Option C: Use one RTSoA type behind a Quokka schema-aware container

Represent every population with a thin Quokka wrapper derived from `amrex::ParticleContainerRTSoA<>`. Give every instance an immutable schema and provide layout-aware creation, access, output, and restart operations.

This eliminates container-type variation while keeping AMReX redistribution and particle I/O infrastructure. It requires a deliberate conversion of all AoS-specific kernels and helpers.

### Option D: Maintain a permanent abstraction over both AoS and RTSoA

Introduce a layout-polymorphic adapter and support both storage models indefinitely.

This could stage the transition, but permanent dual-layout support would multiply test coverage and preserve the very layout branching the migration is intended to remove. A short-lived branch or compile-time transition adapter is acceptable; a permanent public dual-layout API is not.

## Decision

Choose Option C.

Quokka will migrate every physics and tracer particle population to one schema-aware container built on `amrex::ParticleContainerRTSoA<>`. The final architecture will not expose AoS storage or compile-time payload counts to particle algorithms.

### Container ownership

Introduce a concrete `quokka::ParticleContainer` wrapper derived from `amrex::ParticleContainerRTSoA<>`.

The wrapper will:

- own one immutable `ParticleSchema` for its lifetime;
- configure the AMReX arena and runtime components before any tile is created;
- create and define tiles only through schema-aware methods;
- apply instance I/O options to every container;
- enforce empty-container restart semantics;
- expose schema-aware initialization and output entry points; and
- reject unsupported AMReX paths with always-on diagnostics.

Each physical population remains a separate container instance selected by `quokka::ParticleType`. The instances share a C++ container type but not a schema or particle data.

The wrapper uses inheritance rather than composition because Quokka and AMReX algorithms require the particle-container iterator, redistribution, mesh-deposition, and `ParGDB` interfaces directly. The wrapper must remain thin: storage, ownership maps, redistribution, and binary particle I/O remain AMReX responsibilities.

### Particle schema

Introduce a host-side `ParticleSchema` containing:

- the `quokka::ParticleType` population tag;
- ordered logical real-component names;
- ordered logical integer-component names;
- deterministic default values for every component;
- field units or a key into the existing units metadata;
- capability indices such as mass, velocity, birth time, death time, luminosity, and evolution stage; and
- the mapping from logical component indices to RTSoA storage indices.

Schemas are produced by `ParticleSchemaTraits<particle_type, problem_t>` during container construction. They are derived from the existing enums, radiation-group count, and stellar-model component declarations. They are not parsed from runtime inputs.

A schema is frozen before the first tile is defined. Adding or resizing runtime components after construction is prohibited.

Every schema must satisfy these invariants:

- component names are nonempty and unique within their scalar type;
- every capability index is either absent or in range;
- every default vector exactly matches its component-name vector;
- positions occupy RTSoA real storage indices `0..AMREX_SPACEDIM-1`;
- logical real component `i` maps to storage real component `AMREX_SPACEDIM + i`;
- logical integer component `i` maps to storage integer component `i`; and
- AMReX I/O names and masks have the exact lengths required by the container.

Existing public constants such as `CICParticleMassIdx` retain their meaning as logical physics indices. They must never be passed directly to raw RTSoA `rdata` access.

### Future pyAMReX binding boundary

The common container and schema form the supported C++ boundary for future Python bindings.

Bindings should expose:

- a collection of particle populations identified by `quokka::ParticleType` and population name;
- the immutable schema for each population, including component names, scalar types, logical order, and units;
- particle counts and hierarchy metadata;
- component arrays selected by name or validated logical index; and
- explicit synchronization and lifetime rules for any zero-copy host or device view.

Bindings must not expose population-specific C++ template types, raw `ParticleTileDataRT` pointers, or the position-prefixed AMReX storage indices as the public Python data model. Position arrays and logical physics-component arrays remain distinct at the binding layer even though positions occupy the first runtime real components internally.

The schema API must therefore be usable without a `problem_t` template argument after container construction. Python-facing introspection reads metadata from the container instance; it does not reconstruct the schema from Quokka compile-time traits.

This ADR does not choose a Python array protocol or memory-sharing mechanism. NumPy, CUDA/HIP-aware array interfaces, DLPack, and copied host arrays may be evaluated when the bindings are designed, provided they all preserve the schema and lifetime contract above.

### GPU data access

Introduce a trivially copyable `ParticleLayout` value containing only GPU-usable component indices. The host-side schema produces this value after validation.

Hot kernels will obtain `ParticleTileDataRT` from each iterator and access a particle by index. Quokka helpers will provide the semantic operations:

- `position(particle, dir)`;
- `real(particle, logical_index)`;
- `integer(particle, logical_index)`;
- `id(particle)`; and
- `cpu(particle)`.

These helpers apply the real-component position offset and are usable from CPU and GPU code. Physics-specific functors may capture validated storage indices directly when that makes a hot loop clearer.

Quokka code must not:

- call `GetArrayOfStructs()`;
- use `ParticleType::NReal` or `ParticleType::NInt` for payload discovery;
- take or accept `ParticleType*` ranges;
- assume `&rdata(0)` or `&idata(0)` points to all fields of one particle; or
- pass an unchecked signed component index to AMReX.

Stellar-model interfaces that currently accept contiguous real and integer pointers will be changed to accept a particle accessor/view or explicit component values. Particle-creation functors will receive tile data and particle indices rather than pointers to AoS particle arrays.

### Initialization

All initialization enters through the Quokka wrapper so that schema validation, deterministic defaults, ID allocation, and tile definition are uniform.

The required initialization modes are:

- ASCII input for existing problem generators;
- random initialization;
- one-particle-per-cell tracer initialization; and
- GPU particle creation by Quokka physics modules.

The wrapper may delegate to an AMReX initializer only after the corresponding P0 audit bugs are fixed and covered by a pure-RTSoA regression. Otherwise Quokka will use a schema-aware implementation with the same external behavior.

Every initialization path will:

1. validate its supplied fields against the schema;
2. initialize every omitted field to its schema default;
3. allocate each particle ID exactly once;
4. define the destination tile before resizing it;
5. populate complete tile data; and
6. redistribute before returning when particles may belong to another rank or level.

### Physics kernels

Particle algorithms will operate on tile data plus a `ParticleLayout`, not on a container-specific AoS type. Capability checks remain at the host dispatch boundary; hot kernels receive only validated indices.

The conversion includes:

- drift and kick;
- creation and destruction;
- mesh deposition and feedback deposition;
- sink and star accretion;
- particle radiation and stellar evolution;
- particle splitting;
- reductions and statistics; and
- particle gathering for diagnostics.

Complete particle copies must copy ID/CPU, every real component, and every integer component. AoS-only overloads and temporary AoS bridge containers are prohibited.

### Tracer particles

Tracer particles will use the same Quokka RTSoA wrapper and an empty physics schema: positions and ID/CPU are stored, with no user real or integer fields unless tracer features later require them.

Quokka will implement tracer advection against `ParticleTileDataRT` rather than inherit `amrex::AmrTracerParticleContainer`, whose implementation is AoS-specific. Tracer plotfile and checkpoint population names remain `tracer_particles`.

### Output and restart

`PhysicsParticleRegister` remains the population registry and the sole high-level dispatch point for physics-particle output. Its descriptors will refer to the common container type plus a population schema instead of relying on the concrete container template to encode component counts.

Plotfile and checkpoint output will preserve:

- the existing population directory names;
- the existing logical real and integer field names and order;
- the existing units metadata; and
- the existing inclusion of all schema fields in checkpoints.

Quokka will generate component names and masks from `ParticleSchema` immediately before calling AMReX and will validate their exact sizes. Selective checkpoints are prohibited. Selective plotfile output may be introduced separately after it has round-trip-independent tests.

Quokka's text output and diagnostic gather operations will copy layout-independent tile data to host buffers. They will not use AMReX `WriteAsciiFile` or construct a single-box AoS analysis container.

Restart is permitted only into an empty container with a schema compatible with the particle header. Compatibility is defined by scalar type, component count, field order, and field names where the format records them.

After all particle populations have restarted, Quokka will restore the shared RTSoA next-ID state once. The restored value will be at least the maximum checkpoint `next_id` requirement and greater than every valid restored particle ID across all populations and MPI ranks. No particle may be created between the first population restart and this finalization step.

Existing AoS checkpoints are part of the migration test matrix. If AMReX cannot read them directly into an equivalent RTSoA schema, Quokka will provide a one-time compatibility reader without changing newly written checkpoint formats.

### Unsupported AMReX paths

The wrapper will reject or avoid these paths until separately approved:

- asynchronous particle output;
- mutable runtime component layouts;
- AMReX ASCII particle output;
- AoS transfer overloads;
- virtual and ghost particle creation;
- neighbor-particle APIs; and
- restart into a populated container.

The existing Quokka fatal diagnostic for asynchronous output remains in place.

## Consequences

### Benefits

- Every population uses the same container representation and iterator model.
- Radiation-group and stellar-model field-count changes no longer produce new AMReX container types.
- Component schemas become explicit metadata rather than implicit template arguments.
- Particle algorithms can share implementations based on capabilities and validated layouts.
- SoA storage improves component-wise memory access for deposition, updates, and reductions.
- I/O names, defaults, and units acquire one source of truth.
- Future pyAMReX bindings can bind one container API and discover population fields from instance metadata.

### Costs and risks

- This is a cross-cutting refactor of nearly every particle kernel and helper.
- Per-particle proxy access may obscure noncoalesced algorithms; performance must be measured rather than assumed.
- The common container specialization also shares AMReX static configuration and next-ID state, requiring explicit lifecycle control.
- Schema/index mistakes can silently access positions as physics fields unless all raw access is removed.
- Existing problem code that directly manipulates AoS particles must migrate to accessors.
- The stellar-model interface must stop accepting contiguous per-particle component pointers.
- Checkpoint compatibility must be demonstrated for every population, including radiation-group and stellar-model variants.

### Performance expectations

The migration must not be justified by assumed speedups alone. Component-wise kernels are expected to benefit from SoA access, while algorithms that touch every field may see neutral or worse locality.

Before acceptance, benchmark representative CPU, CUDA, and HIP configurations for:

- drift and kick;
- CIC mass deposition;
- sink accretion;
- radiation luminosity updates;
- particle creation and redistribution; and
- checkpoint write and restart.

Regressions greater than 5% in a representative particle-dominated kernel require investigation and documentation. This threshold is a review trigger, not an automatic rejection when a justified correctness or compile-time benefit outweighs it.

### Interfaces and formats

- No `ParmParse` option is added, removed, or renamed.
- Particle checkpoint and plotfile formats must remain compatible.
- Problem generators may require source changes when they directly call AoS APIs, but their physical and runtime-input behavior must remain unchanged.
- The existing population enum and logical component-index constants remain supported during migration.
- The future Python API will use logical component names and indices, not raw position-prefixed RTSoA storage indices.

## Rollout & Testing

The migration will proceed as vertical slices. A population advances only after initialization, evolution, output, and restart all work with RTSoA. The tree must not remain indefinitely in a partially abstracted dual-layout state.

### Milestone 0: AMReX readiness

- Fix the P0 bugs in the AMReX migration triage.
- Add an upstream pure-RTSoA initialization and checkpoint/restart test.
- Update Quokka's AMReX revision.
- Record the exact AMReX commit satisfying the gate in this ADR or its implementation PR.

### Milestone 1: Quokka infrastructure

- Add `ParticleSchema`, `ParticleSchemaTraits`, and GPU-safe `ParticleLayout`.
- Add the `quokka::ParticleContainer` wrapper and schema-aware tile definition.
- Add typed particle access helpers.
- Add schema validation, deterministic defaults, and per-instance I/O configuration.
- Add a small unit test covering schema construction and logical-to-storage index mapping.
- Add a non-templated schema-introspection test exercising two populations with different runtime layouts; this is the contract future pyAMReX bindings will consume.

### Milestone 2: first end-to-end population

- Convert the test-particle container first because it exercises real fields, integer fields, creation, output, and restart without being a production-only schema.
- Convert its kernels and problem tests without compatibility shims in hot loops.
- Compare AoS and RTSoA particle records after initialization and one evolution step.

### Milestone 3: gravitating and radiating populations

- Convert CIC particles.
- Convert radiation particles.
- Convert combined CIC-radiation particles.
- Verify deposition, drift/kick, radiation updates, and multilevel redistribution.

### Milestone 4: sink and stellar populations

- Convert sink particles and accretion.
- Refactor stellar-model access away from contiguous component pointers.
- Convert star and stochastic stellar-population particles.
- Convert creation, destruction, splitting, feedback deposition, and model-specific extra fields.

### Milestone 5: tracers and common I/O

- Implement RTSoA tracer initialization and advection.
- Convert diagnostic gathers, statistics, and text output.
- Remove `amrex::AmrTracerParticleContainer` and the remaining AoS-only helpers.
- Remove transitional container aliases and layout branches.

### Required test matrix

For each population, test:

- zero particles and nonzero particles;
- one and multiple AMR levels;
- one and multiple MPI ranks;
- CPU and at least CUDA or HIP in CI, with both GPU backends tested before final acceptance;
- creation or initialization followed by redistribution;
- particle update/deposition behavior;
- plotfile output;
- checkpoint and restart; and
- restart from a representative pre-migration AoS checkpoint.

Cross-population tests must verify:

- two or more simultaneous RTSoA containers with different schemas;
- correct per-instance I/O options;
- preserved field names, order, values, and units;
- no ID collisions after multi-population restart;
- deterministic initialization of omitted fields; and
- no direct AoS access remaining under `src/particles/` or tracer paths.

Numerical regression tests retain their current tolerances unless the implementation PR documents and justifies a change. Particle counts, IDs, schema metadata, and checkpoint round trips require exact agreement.

### Review roles

- **Driver:** Quokka particle maintainers.
- **Required reviewers:** one AMReX particle maintainer and one Quokka GPU maintainer.
- **Consulted:** maintainers of sink/star physics, radiation particles, and checkpoint compatibility.

The ADR remains Proposed until the AMReX readiness milestone and the first end-to-end RTSoA population demonstrate that the design is viable on CPU and GPU.

## Links

- [Particle Roadmap in AMReX/pyAMReX](https://github.com/AMReX-Codes/pyamrex/issues/460)
- [AMReX particle bugs affecting the RTSoA migration](../markdown/amrex_particle_rtsoa_migration_bug_triage.md)
- [`amrex::ParticleContainerRTSoA<>` definition](../../extern/amrex/Src/Particle/AMReX_ParticleContainer.H)
- [`ParticleTileDataRT` and `RTSoAParticle`](../../extern/amrex/Src/Particle/AMReX_ParticleTileRT.H)
- [Current particle type aliases](../../src/particles/particle_types.hpp)
- [Current particle creation implementation](../../src/particles/particle_creation.hpp)
- [Current particle I/O helpers](../../src/particles/particle_IO.hpp)
- [Current physics-particle registry](../../src/particles/PhysicsParticles.hpp)
- [ADR process](../markdown/adrs.md)
