# ADR-0001: Runtime problem definitions and Python-driven solvers

Date: 2026-07-18 • Status: Proposed

Decision scope: problem generators, runtime physics composition, solver organization, Python bindings, and chemistry integration ownership

Decision owners: Quokka maintainers

Review triggers: a prototype invalidates the performance assumptions; GPU toolchains cannot support the network/module strategy; or the accepted compilation policy changes

## Executive summary

Quokka can be redesigned so that authoring a new problem, changing a radiation or dust group count, changing passive scalars, selecting existing particle processes, or composing existing physics operators does not require C++ compilation.

After an execution bundle has been built for a spatial dimension, CPU/GPU backend, precision, MPI environment, and chemistry network, compilation should be required only to add executable physics:

- a new equation of state (EOS) implementation;
- a new opacity implementation;
- a new chemistry network;
- a completely new physics operator; or
- optionally, a new optimized native kernel for behavior that already has a slower interpreted path.

The architecture required to achieve that is stronger than replacing problem specializations with a finite catalog of template instantiations. The production target should be:

1. a non-templated `QuokkaSimulation` and AMR state whose field layout is assembled at runtime;
2. a runtime graph of compiled physics operators selected from registries;
3. runtime EOS and opacity selection from the compiled model catalog;
4. runtime radiation groups, dust groups, passive scalars, and particle components;
5. a validated `ProblemPlan` for initial conditions, boundaries, refinement, sources, and diagnostics;
6. a non-template `Simulation` interface shared by a generic C++ driver and Python bindings; and
7. a WarpX-style Python binding stack: a narrow custom pybind11 module for Quokka concepts layered with pyAMReX for low-level AMReX types and data access; and
8. C++/CUDA/HIP execution for numerical kernels, with Python used for composition and orchestration rather than per-cell callbacks.

Templates can remain inside numerical implementations for directions, local algorithms, or model-specific optimization. They must stop propagating problem identity and runtime configuration through the simulation type.

A prerequisite is to remove the production dependency on Microphysics and replace it with a narrowly vendored Rosenbrock integrator plus Quokka-owned EOS and chemistry-network interfaces. Microphysics is not fully templated over the selected network and integrator state: generated headers, global parameters, and concrete state types leak the build-time network choice into its consumers. That prevents the solver from being a reusable runtime core with clean, separately compiled network modules. The redesigned chemistry path must share one network-independent Rosenbrock implementation while treating each new RHS/Jacobian network as the compiled unit.

The AMReX/pyAMReX particle roadmap supports the same runtime direction, but it also makes Quokka's current particle layout a migration blocker. Quokka must leave the legacy compile-time AoS `AmrParticleContainer<NReal, NInt>` family, use one generic polymorphic PureSoA storage type with runtime fields, and isolate that storage behind a Quokka `ParticleStore` interface. The already-bound PureSoA type is a bridge; AMReX's newer runtime-only `ParticleContainerRTSoA` is the target after pyAMReX exposes it and Quokka verifies feature parity.

The intended end state is:

```text
Python problem script or generic C++ driver
                    |
                    v
          non-template Simulation interface
                    |
                    v
       validated runtime configuration
                    |
          +---------+----------+
          |                    |
          v                    v
    runtime StateSchema    runtime OperatorGraph
                               |
                    +----------+-----------+
                    |          |           |
                  Hydro     Radiation   Gravity ...
                    |          |
                    v          v
             compiled EOS  compiled opacity
               registry       registry
                    |
                    v
             AMReX / MPI / CUDA / HIP
```

Runtime dispatch happens at configuration or kernel launch. No Python call, string lookup, virtual dispatch, or dynamic allocation should occur once per cell.

## Compilation policy

The policy should be stated in terms of a previously built execution bundle.

### Must not require compilation

- a new problem script using existing physics;
- enabling or disabling an existing operator;
- changing hydro, radiation, MHD, gravity, dust, conduction, cooling, or existing particle options;
- changing passive-scalar or mass-scalar counts;
- changing radiation or dust group counts and group boundaries;
- selecting an existing EOS or opacity and changing its parameters;
- selecting existing particle families or changing their runtime fields;
- changing initial conditions, boundaries, refinement criteria, source parameters, or diagnostics;
- changing the input deck; or
- adding Python-side validation and analysis.

### May require compilation

- implementing a new EOS formula or backend;
- implementing a new opacity formula or backend;
- generating or adding a new chemistry network;
- implementing a new physics operator or a new particle evolution law;
- adding a new native GPU kernel when the expression/stencil facilities are insufficient or too slow; and
- building for a different `AMREX_SPACEDIM`, CPU/CUDA/HIP backend, precision, MPI ABI, or platform.

The final item is a platform build constraint, not a problem-generator constraint. A user should still be able to run many runtime-defined problems from the resulting bundle.

### Functional path versus optimized path

The no-compilation rule should be functional, not merely cover a fixed menu of example problems. Runtime problem facilities need:

- an expression engine for coordinate-, time-, and state-dependent values;
- composable masks and piecewise expressions;
- generic stencil taggers and field transforms;
- explicit host-copy or Python patch callbacks as a slow escape hatch; and
- registered optimized launchers for common production patterns.

A problem that can be expressed through those facilities needs no native code. Users may optionally add an optimized launcher, but lack of one must not make compilation mandatory.

## Why the current design prevents this

### Problem identity has too many responsibilities

The current design gives each problem an empty tag type and uses that type to select:

- `Physics_Traits<problem_t>` and the state-vector size;
- `EOS_Traits<problem_t>`, `HydroSystem_Traits<problem_t>`, and `RadSystem_Traits<problem_t>`;
- particle container component counts;
- `SimulationData<problem_t>`;
- initial conditions, boundary conditions, refinement, sources, and diagnostics;
- the concrete `QuokkaSimulation<problem_t>` type;
- `problem_main()`; and
- a separate CMake executable target.

The pattern is documented in `developing_problem_generators.md` and is visible in `src/problems/HydroShocktube/testHydroShocktube.cpp`. A small shock tube needs an empty tag, multiple traits, multiple member specializations, and a process entry point even though its real inputs are a few primitive states and thresholds.

At the time of this audit, `src/problems/` contains:

- 105 problem driver translation units;
- 836 occurrences of explicit `template <>` specialization;
- 113 `QuokkaSimulation<...>::setInitialConditionsOnGrid` specializations;
- 116 `Physics_Traits` and 113 `EOS_Traits` specializations; and
- approximately 37,600 lines across problem driver translation units.

Not every template in those files is unnecessary. The counts show that the problem type has become a global routing key rather than a narrow numerical abstraction.

### Compile-time choices are mixed together

Some current compile-time choices genuinely select executable physics, but many are ordinary runtime dimensions or parameters:

| Current compile-time choice | Required target |
|---|---|
| Hydro/radiation/MHD/dust/gravity enabled flags | Runtime operator selection |
| Passive and mass scalar counts | Runtime state-schema fields |
| Radiation and dust group counts | Runtime group ranges |
| Particle switches and extra components | Runtime particle schemas and operator selection |
| EOS parameters such as gamma and molecular weight | Runtime model parameters |
| Existing EOS backend selection | Runtime model registry |
| Existing opacity model selection and group boundaries | Runtime model registry and arrays |
| Problem initial conditions and hooks | Runtime `ProblemPlan` |
| New EOS/opacity implementation | Compiled model module |
| New chemistry network | Compiled execution bundle or network module |
| Existing chemistry network and Rosenbrock parameters | Runtime selection/configuration within a compatible bundle |
| New physics operator | Compiled operator module |

The migration succeeds only if the left-hand runtime rows stop participating in C++ types.

### Existing virtual hooks are not a runtime problem seam

`AMRSimulation<problem_t>` already declares virtual methods, but `QuokkaSimulation` remains templated, state indices remain static, boundary functions are specialized device functions, and problems specialize member definitions rather than supplying an object through one interface.

Turning those methods into one large bindable `Problem` base class would still be a shallow interface. It would also invite Python overrides at device-scale call sites. The replacement needs data-driven plans and coarse launch interfaces, not a mechanical virtual version of every specialization.

### Existing runtime precedents

Quokka already has useful local precedents:

- `src/Factory.H` performs runtime selection of registered adapters.
- `DiagBase` and `DerivedFieldBase` are selected at runtime.
- `DerivedFieldBase::ComputeContext` passes constrained capabilities rather than a concrete simulation type.
- AMReX `MultiFab` component counts are supplied at runtime.
- the vendored AMReX particle container supports named runtime real and integer SoA components through `AddRealComp` and `AddIntComp`, and it includes the newer runtime-only `ParticleContainerRTSoA` implementation.
- AMReX's parser produces host/device `ParserExecutor` objects from runtime expressions.

These facilities mean the desired architecture is feasible without replacing AMReX.

## Architectural principles

### Runtime configuration describes data; compiled modules provide behavior

A configuration may say:

- use the existing ideal-gas EOS with gamma 1.4;
- use five radiation groups with these boundaries;
- enable hydro, multigroup radiation, self-gravity, and two particle processes;
- initialize fields from these expressions and a table;
- tag cells using these stencil criteria; and
- run these diagnostics.

None of those choices adds executable physics. They should be data.

A new EOS formula, opacity law, chemistry network, chemistry integrator, or time-advance operator adds executable behavior. It belongs in a compiled module registered under a stable name. The normal chemistry path uses Quokka's vendored Rosenbrock integrator; adding another integration algorithm is therefore a new physics operator, while changing tolerances or selecting an already compiled network remains runtime configuration.

### State layout is assembled, then frozen

The enabled operator set declares its required fields into a `StateSchemaBuilder`. The builder resolves names, centering, component ranges, group structure, ghost requirements, and restart metadata. The schema is validated and frozen before AMR allocation.

After freezing:

- fields cannot be added or reordered;
- launchers resolve names to integer `FieldId`/range values once;
- device kernels receive integers and lightweight views, not strings;
- checkpoint metadata records a schema identifier and full field table; and
- a restart must match or pass an explicit migration.

### Dispatch belongs outside cell loops

There are three acceptable dispatch locations:

1. once while assembling the operator graph;
2. once when selecting a model-specific kernel launcher; or
3. one uniform device-side tag branch when avoiding a compile-time model cross product.

There must not be one polymorphic or Python call per cell, group, face, or particle.

Runtime field layout must also not imply repeated runtime index derivation inside hot loops. When the schema and operator graph are frozen, each launcher should resolve its named fields and ranges into a compact, trivially copyable index pack. Base component offsets, range endpoints, strides, and any fixed relative offsets should be computed before kernel launch. A kernel may add a loop variable to a precomputed base offset, but it should not repeatedly search metadata, compose offsets through several abstractions, or recompute the same component expression per cell.

This is not a fundamental obstacle to runtime schemas: a runtime integer offset passed as launch data can be as cheap as other kernel parameters. It is nevertheless a required performance check because extra address arithmetic, missed inlining, oversized index packs, or increased register pressure can accidentally make the runtime implementation slower than the current statically indexed path.

### Python drives; C++ owns

Python owns composition, parameterization, control flow, validation, and analysis. C++ owns:

- AMReX lifetime;
- distributed AMR state;
- device memory;
- kernel launches;
- MPI collectives;
- operator ordering and invariants; and
- checkpoint/plotfile compatibility.

## Proposed target modules

### 1. `Runtime`

`Runtime` owns AMReX initialization and finalization. Importing the Python extension must not initialize AMReX implicitly.

Requirements:

- one active runtime per process initially;
- the runtime outlives simulations and state views;
- support for the current input/TOML path and explicit overrides;
- destruction of all AMReX-dependent C++ objects before finalization;
- execution-bundle metadata including dimension, backend, precision, MPI, and chemistry network; and
- Python context-manager support.

The supported MPI launch model should initially be:

```console
mpiexec -n 4 python problem.py inputs/problem.toml
```

Every rank executes the same script and constructs the same configuration.

### 2. `RunConfig` and `PhysicsConfig`

`RunConfig` owns AMR, timestep, I/O, and run-control options. `PhysicsConfig` selects existing operators and model instances.

Both are typed values, not unvalidated string dictionaries. They can load existing TOML/`ParmParse` data and apply Python keyword overrides. Validation happens before allocation.

An illustrative `PhysicsConfig` could contain:

```cpp
struct PhysicsConfig {
    std::optional<HydroConfig> hydro;
    std::optional<RadiationConfig> radiation;
    std::optional<MhdConfig> mhd;
    std::optional<DustConfig> dust;
    std::optional<GravityConfig> gravity;
    std::optional<ConductionConfig> conduction;
    std::vector<ParticleProcessConfig> particle_processes;
    ScalarConfig scalars;
};
```

An absent configuration disables an operator. Enabling a compiled operator does not alter the simulation type.

### 3. `StateSchema`, `FieldId`, and runtime views

`StateSchema` replaces `Physics_Indices<problem_t>` at the public seam.

It describes:

- cell-, face-, edge-, node-, and particle-centered fields;
- scalar component indices and contiguous ranges;
- group axes and group boundaries;
- conserved/primitive semantics;
- names and units;
- ghost-cell requirements;
- output and restart policy; and
- schema version.

Example:

```cpp
auto density = schema.require_scalar("gasDensity");
auto momentum = schema.require_range("gasMomentum", AMREX_SPACEDIM);
auto radiation = schema.require_group_range("radiation", config.n_groups);
auto passive = schema.require_range("passiveScalars", config.n_passive_scalars);
```

`MultiFab::define(..., ncomp, ngrow)` already accepts runtime component counts, so cell and face state do not require a compile-time problem layout.

Device kernels receive compact values such as:

```cpp
struct HydroFields {
    int density;
    int momentum_first;
    int total_energy;
    int internal_energy;
    int passive_first;
    int passive_count;
};
```

These are resolved once and copied by value into a launch. No string lookup occurs on device.

### 4. `OperatorGraph`

The simulation contains a runtime graph of compiled physics operators. Existing operator kinds include:

- hyperbolic hydro;
- radiation transport and matter-radiation coupling;
- MHD and resistivity;
- dust transport and drag;
- self-gravity;
- conduction;
- cooling and chemistry;
- turbulence driving;
- particle creation, deposition, feedback, and evolution; and
- diagnostics and output transforms.

The graph captures explicit ordering and dependencies rather than hiding order in `if constexpr` blocks throughout a monolithic class.

A single enormous `PhysicsOperator` interface would be shallow. Prefer a small set of role interfaces or phase launchers:

```cpp
struct OperatorDescriptor {
    std::string name;
    std::vector<FieldRequirement> fields;
    std::vector<PhaseLauncher> phases;
    std::vector<OrderConstraint> ordering;
};
```

Named phases can include:

- `prepare`;
- `estimate_timestep`;
- `pre_advance`;
- `hyperbolic_advance`;
- `strang_source_forward`;
- `strang_source_reverse`;
- `elliptic_solve`;
- `particle_update`;
- `post_advance`; and
- `diagnostics`.

The scheduler validates that dependencies exist and constructs an immutable execution order. Adding a new operator requires compilation; selecting any registered operator is runtime configuration.

### 5. EOS registry

An EOS module supplies a compiled implementation and declares:

- its stable name and version;
- required parameters and scalar/species fields;
- host/device capabilities;
- thermodynamic operations;
- validity domain; and
- serialization metadata.

Selecting `ideal_gas` versus `tabulated` is runtime. Adding a third EOS implementation requires compilation.

Hot-loop dispatch has two viable implementations:

1. **Host launch dispatch:** visit the selected EOS once and launch an EOS-specialized hydro kernel.
2. **Uniform device tag:** copy a compact discriminated EOS data object to the device and switch on its model tag inside an inline function.

Host dispatch offers maximum optimization but compiles hydro launch variants for each EOS. That is acceptable because adding an EOS is explicitly a compilation event. A uniform tag avoids a cross product when several operators use the EOS. Benchmarks should choose per call path.

Runtime EOS parameters—gamma, molecular weight, table handle, floors, and unit conversions—must live in the model data rather than template traits.

Mass-scalar inputs should use a runtime `DeviceSpan`/field range. Chemistry networks may still expose a fixed generated species schema inside their compiled network module.

### 6. Opacity registry

Opacity follows the same pattern:

- constant, piecewise, tabulated, or other existing models are runtime selections;
- coefficients, tables, and group boundaries are runtime data;
- a new opacity formula requires compilation; and
- radiation kernels dispatch at launch or through a uniform device model tag.

Opacity must not be specialized by problem. A problem selects a named opacity instance and supplies its parameters.

### 7. Vendored Rosenbrock integrator and chemistry network modules

Microphysics must not remain underneath the new runtime interface. It is not fully templated over its network and integrator state, and its generated unqualified headers, global parameter storage, `burn_t`/`eos_t`-style concrete types, and target-wide build configuration make the chosen network visible throughout Quokka. Wrapping that interface would hide the dependency syntactically without creating an independently reusable solver.

Quokka should instead vendor the minimum Rosenbrock implementation needed by its chemistry operators. The vendored module owns:

- the Rosenbrock tableau and stage construction;
- adaptive step-size selection, error norms, rejection, and retry behavior;
- validity and endpoint cleanup protocol;
- small dense linear solves, including an optimized six-equation path where it remains beneficial;
- CPU/CUDA/HIP-compatible workspace types; and
- structured success/failure diagnostics.

The vendor directory must retain the upstream license, exact source revision, a record of local patches, and a repeatable update procedure. Quokka-specific network adaptation belongs outside the vendored source so future upstream comparisons remain possible.

Quokka then owns a compiled `ChemistryNetwork` interface that supplies:

- stable network name/version and species metadata;
- immutable device network data and runtime parameters;
- mapping between named Quokka fields and the integrator vector;
- RHS and analytic Jacobian evaluation;
- error-control participation and scaling for each integrated variable;
- state validity, conservation, charge-closure, and positivity rules; and
- gas/radiation energy-coupling results.

The distinction between integrated state and error-controlled state must be explicit. In the current photoionization path, for example, radiation flux is evolved in the network but is passive with respect to convergence, validity, and timestep rejection. The new interface must be able to express that without special knowledge inside the generic Rosenbrock code.

EOS operations used during a chemistry update go through the EOS registry; chemistry code must not expose Microphysics EOS types. Generated species layout may remain fixed inside a compiled network module, but no generated header or network-specific type may enter `Simulation`, the operator graph, or the Python binding.

The runtime interface remains uniform:

```cpp
auto chemistry = model_registry.create_chemistry("primordial_chem", parameters);
physics.enable(chemistry);
```

Changing network parameters or enabling a network already present in a bundle is runtime. Generating and adding a new network requires compilation.

Initially it is acceptable to ship one network per execution bundle. A later design may load multiple network shared libraries, but portable CUDA/HIP device linking and ABI stability should be proven before making dynamic plugins a requirement.

Migration is complete only when production Quokka targets no longer include or link `extern/Microphysics`, consume its generated headers or global parameter objects, call its CMake setup helpers, or depend on its state types. VODE need not be carried into the replacement unless a separate requirement justifies it; the supported built-in chemistry integrator is Rosenbrock.

### 8. Runtime radiation and dust groups

Radiation and dust group counts must not define C++ types.

The current multigroup implementation uses many `amrex::GpuArray<T, nGroups>` and `quokka::valarray<T, nGroups>` values, including local nonlinear-solve scratch. Replacing those is the largest numerical refactor in this design.

The target uses:

- a runtime `GroupRange { first_component, count, stride }`;
- device spans into state and model tables;
- loops bounded by runtime `count`;
- preallocated workspace `MultiFab` or per-tile scratch for values that cannot stream directly from state;
- algorithms that avoid per-cell heap allocation; and
- an optional optimized fast path for common small counts.

The generic path must support a new group count without rebuilding. An optimized path may specialize `N=1` or other common counts internally, but a missing fast path must fall back to the generic runtime loop.

The matter-radiation nonlinear solve needs particular care. Fixed-size Jacobian and residual types should become a runtime workspace. Where the Jacobian has diagonal/arrowhead structure, the algorithm should exploit that structure so scratch grows as `O(n_groups)` rather than allocating a dense matrix per cell.

Dust drag and charge models need the same conversion from fixed `GpuArray<nDustGroups>` values to runtime spans and workspace.

### 9. Runtime scalar fields

Passive and mass scalars are ordinary contiguous field ranges:

```cpp
struct ScalarRange {
    int first;
    int count;
    int mass_scalar_count;
};
```

Hydro reconstruction, flux updates, interpolation, fixups, deposition, and I/O loop over the runtime count. EOS and chemistry receive views rather than fixed-size `GpuArray` values.

Optional specialized loops for very small counts may exist as performance fast paths, but the generic path guarantees that changing the count never requires compilation.

### 10. Runtime particle schemas

Quokka currently encodes several particle component counts in aliases such as `AmrParticleContainer<NReal, NInt>`. This is not merely a compile-time naming issue: that alias instantiates the legacy mixed-layout `Particle<NReal, NInt>` representation, and current Quokka kernels, I/O, creation, destruction, deposition, and restart code access compile-time `NReal`/`NInt` values and `ArrayOfStructs` storage directly.

The [AMReX/pyAMReX particle roadmap](https://github.com/AMReX-Codes/pyamrex/issues/460) makes the intended upstream direction explicit:

1. new applications should use PureSoA, and pyAMReX intends to [drop legacy AoS support](https://github.com/AMReX-Codes/pyamrex/issues/459);
2. particle containers should use `PolymorphicArenaAllocator`, eliminating separate host, pinned, managed, and device container types; and
3. all AMReX applications should converge on one PureSoA particle type whose position and application fields are runtime components.

The roadmap is partly implemented as of this decision. Polymorphic particle-container bindings landed in [pyAMReX PR 428](https://github.com/AMReX-Codes/pyamrex/pull/428), and WarpX migrated to them in [WarpX PR 6374](https://github.com/BLAST-WarpX/warpx/pull/6374). AMReX PR [4404](https://github.com/AMReX-Codes/amrex/pull/4404) added `amrex::ParticleContainerRTSoA`, a runtime-only, component-major two-dimensional particle store using `amrex::PolymorphicArenaAllocator`; it is already present in Quokka's inspected AMReX revision. That PR's HiPACE++ measurements were performance-neutral relative to a fixed-component PureSoA container. However, inspected pyAMReX commit [`36a4576`](https://github.com/AMReX-Codes/pyamrex/tree/36a4576b90ccf99d0daf78e3aeba3b4f7f1b613b) does not yet bind `amrex::ParticleContainerRTSoA`; its generic binding is still a polymorphic `amrex::ParticleContainerPureSoA<AMREX_SPACEDIM, 0>` with positions fixed by dimension and application attributes added at runtime.

Quokka should follow the roadmap without coupling the public interface to its delivery schedule. Introduce one deep `ParticleStore` module whose interface owns:

- named logical particle populations;
- a frozen `ParticleSchema` for each population;
- AMR grid metadata and redistribution;
- arena selection and allocation state;
- checkpoint/restart schema metadata; and
- creation of operator-specific launch data.

The concrete AMReX container is an implementation detail behind that seam. Multiple logical populations may initially use separate container instances because they have different lifecycle, output, or redistribution rules, but every instance should have the same concrete C++ storage type. Do not encode `Rad`, `CIC`, `Sink`, `Star`, group count, stellar model, or problem identity in the container type.

Use this storage sequence:

1. **Migration bridge:** `amrex::ParticleContainerPureSoA<AMREX_SPACEDIM, 0, amrex::PolymorphicArenaAllocator>` with only positions as compile-time components and every Quokka field as a named runtime component. This type is already supported by the inspected pyAMReX bindings.
2. **Target:** `amrex::ParticleContainerRTSoA<>`, or its stable upstream successor, after Quokka's required AMR, deposition, I/O, restart, and Python-binding capabilities are available. Switching storage implementations must not change the `ParticleStore`, `ParticleSchema`, operator, checkpoint-schema, or Python interfaces.

Do not add `pyAMReX_CODES=Quokka` specializations for each Quokka particle layout. That would reintroduce compilation for group counts, particle fields, and stellar models through the binding build even if the solver itself were dynamic. If pyAMReX still lacks the generic runtime-only binding when Quokka needs it, contribute that one generic binding upstream or maintain a temporary thin adapter for the single generic type.

The target should create each logical population from a runtime `ParticleSchema` before particles are created:

```cpp
ParticleSchema stars;
stars.add_real("mass");
stars.add_real("birth_time");
stars.add_group_real("luminosity", radiation.n_groups);
stars.add_int("evolution_stage");
```

Existing particle processes resolve fields by name during configuration and launch kernels using integer indices. Enabling CIC, sink, stochastic stellar population, or other existing behavior becomes operator selection.

The schema reserves position, particle ID, and owning-rank semantics. It also records component type, ordered name, communication flag, checkpoint flag, units, and default initialization. The memory arena must be selected before the first particle allocation and cannot be changed while particles exist; an explicit copy/migration operation is a separate lifecycle transition. These invariants are validated when the schema is frozen rather than left to individual kernels or Python callers.

Runtime fields must not imply runtime name lookup or repeated component arithmetic inside particle kernels. For each tile, `ParticleStore` resolves names on the host and builds a trivially copyable, operator-specific launch view containing particle count, capacity, and direct pointers to the required position, real, integer, and ID data. Kernels index those pointers by particle number. They should not repeatedly call a name lookup, derive `component * capacity`, or branch on field presence per particle. This follows the roadmap's stated pointer-unwrapping compute pattern and is a particle-specific application of the runtime-index performance rule.

A new particle evolution law is a new physics operator and requires compilation. A new field used by existing generic deposition/advection behavior does not.

### 11. `ProblemPlan`

`ProblemPlan` describes the scenario rather than the physics implementation:

```cpp
struct ProblemPlan {
    InitialConditionPlan initial_conditions;
    BoundaryPlan boundaries;
    RefinementPlan refinement;
    ExternalForcingPlan forcing;
    DerivedFieldPlan derived_fields;
    DiagnosticPlan diagnostics;
    std::shared_ptr<ProblemLifecycle> lifecycle;
};
```

The plan is validated against the frozen state schema and operator graph. It is then immutable except for explicitly owned lifecycle data.

#### Expression fields

An expression initializer can assign primitive or conserved fields as functions of:

- coordinates;
- time;
- runtime constants;
- existing state fields;
- table lookups; and
- piecewise masks.

AMReX `ParserExecutor` provides a host/device-capable starting point. A higher-level builder should validate primitive-to-conserved conversion and field dependencies.

#### Generic stencil operations

Refinement and derived fields often need neighbor values. Provide compiled generic stencil operators parameterized at runtime:

- absolute or relative gradient;
- second derivative;
- divergence and curl;
- threshold and region masks;
- min/max over a stencil;
- distance from geometric primitives; and
- logical combinations.

These cover most existing tagging specializations without compiling a problem.

#### Slow universal escape hatch

For a novel problem operation that is not executable physics, expose an explicit patch-copy interface:

1. copy or map one local patch to a host array;
2. call Python once per patch at an initialization, refinement, or diagnostic phase;
3. validate the returned shape and fields; and
4. copy results back if mutation is allowed.

This is not a production GPU path, but it preserves the rule that a problem never *must* be compiled. A reusable native launcher or future JIT path can optimize it later.

### 12. Registered problem launchers

Common high-performance patterns should be compiled once and selected from Python:

- uniform and piecewise-uniform initialization;
- shock tubes;
- Gaussian/spherical profiles;
- table-interpolated profiles;
- primitive-state expression initialization;
- constant, reflecting, outflow, and expression boundaries;
- gradient and geometry taggers;
- forcing profiles; and
- common reductions and derived fields.

A launcher contains:

- a stable name/version;
- typed runtime parameters;
- required fields and operators;
- dimension/backend compatibility;
- validation; and
- a host function that submits an AMReX kernel.

The registry in `Factory.H` is a useful precedent, but the new registry needs typed construction, duplicate checks, compatibility metadata, and binding-safe errors.

### 13. `ProblemLifecycle`

Host-side stateful work belongs in a small lifecycle adapter:

```cpp
class ProblemLifecycle {
  public:
    virtual ~ProblemLifecycle() = default;
    virtual void prepare(ProblemBuildContext &) {}
    virtual void before_step(StepContext &) {}
    virtual void after_step(StepContext &) {}
    virtual void finalize(FinalContext &) {}
};
```

Use cases include loading tables, preparing parser executors, uploading buffers, recording a reduction, and running final validation.

Contexts expose limited capabilities, not the concrete simulation class. This follows the direction already visible in `DerivedFieldBase::ComputeContext` and avoids untyped simulation access.

Python may implement lifecycle callbacks only at coarse synchronization points. Callback scope must be declared as all ranks, I/O rank, or collective.

### 14. Non-template `Simulation` interface

The public C++ interface should be small:

```cpp
class Simulation {
  public:
    static auto create(Runtime &, RunConfig, PhysicsConfig, ProblemPlan)
        -> std::unique_ptr<Simulation>;

    void initialize();
    void step(int count = 1);
    void run();
    void write_plotfile();
    void write_checkpoint();

    [[nodiscard]] auto time() const -> double;
    [[nodiscard]] auto step_number() const -> std::int64_t;
    [[nodiscard]] auto state_schema() const -> StateSchema const &;
    [[nodiscard]] auto statistics() const -> StatisticsSnapshot;
};
```

Unlike a type-erased facade over many `QuokkaSimulation<Shape>` instantiations, the implementation should ultimately own one non-templated AMR engine, runtime state schema, and runtime operator graph.

The interface enforces:

```text
configured -> schema frozen -> initialized -> stepping/running -> finished
```

Shape-changing mutation after schema freeze is an error. Ordinary runtime parameters explicitly marked mutable may be updated at synchronization points.

## Python-facing interface

### Binding technology decision

Use a WarpX-style layered pybind11 architecture:

- a custom `_quokka_core` pybind11 module binds the stable Quokka interface (`Runtime`, `PhysicsConfig`, `ProblemPlan`, `StateSchema`, `OperatorGraph`, `Simulation`, and diagnostics); and
- pyAMReX supplies the dimension-specific bindings for low-level AMReX types, including `MultiFab`, geometry, iteration, and particle data.

pyAMReX is therefore part of the supported Python binding stack, but it is not the Quokka solver interface by itself. The C++ runtime continues to own AMR state, regridding, scheduling, MPI collectives, and numerical invariants. Python drives the high-level solver through Quokka-owned objects and may opt into pyAMReX-backed data access for analysis, steering, and advanced workflows.

This conclusion is based on pyAMReX development commit [`36a4576b90ccf99d0daf78e3aeba3b4f7f1b613b`](https://github.com/AMReX-Codes/pyamrex/tree/36a4576b90ccf99d0daf78e3aeba3b4f7f1b613b) and WarpX commit [`2e5f0a26691c5378f75bd337a34d2b5bd5653d48`](https://github.com/BLAST-WarpX/warpx/tree/2e5f0a26691c5378f75bd337a34d2b5bd5653d48), inspected on 2026-07-18, against Quokka's then-vendored AMReX commit `daa00fcb19df7d4bb7263426e41b223e36081714`.

pyAMReX is capable and useful at the AMReX seam:

- it exposes `MultiFab`, `Array4`, `MFIter`, geometry, particle containers, `ParmParse`, and many other AMReX types;
- it exposes `AmrCore` with Python overrides for level creation, remake, cleanup, and error estimation;
- it provides explicit `initialize()`/`finalize()` functions;
- it supports CPU `__array_interface__` and CUDA `__cuda_array_interface__` zero-copy views; and
- it supports MPI and selectable GPU-backend builds.

Those capabilities make pyAMReX useful not only for standalone prototypes, but also as the AMReX data plane beneath an application's own Python interface. Its documented use cases include enhancing an existing AMReX application and writing standalone Python AMR prototypes. See the [pyAMReX project](https://github.com/AMReX-Codes/pyamrex), [Python interface](https://pyamrex.readthedocs.io/en/latest/usage/api.html), and [implementation notes](https://pyamrex.readthedocs.io/en/latest/developers/implementation.html).

WarpX demonstrates the application pattern needed by Quokka:

1. WarpX builds one custom pybind11 extension per compiled geometry and links it to the corresponding WarpX library.
2. Its superbuild forces pyAMReX and WarpX to use the same AMReX source and the same pybind11 dependency. It also asks pyAMReX to precompile WarpX's particle-container layouts. See WarpX's [`pyAMReX.cmake`](https://github.com/BLAST-WarpX/warpx/blob/2e5f0a26691c5378f75bd337a34d2b5bd5653d48/cmake/dependencies/pyAMReX.cmake). That specialization is the current WarpX integration mechanism, not Quokka's target; the runtime-only roadmap removes the need for code-specific particle shapes.
3. The Python loader imports the matching `amrex.space1d`, `amrex.space2d`, or `amrex.space3d` module before importing the WarpX extension. This registers pyAMReX's AMReX types in the shared pybind11 registry before the application module returns those types. See WarpX's [`_libwarpx.py`](https://github.com/BLAST-WarpX/warpx/blob/2e5f0a26691c5378f75bd337a34d2b5bd5653d48/Python/pywarpx/_libwarpx.py).
4. WarpX binds application-owned objects, registries, callbacks, and lifecycle operations itself while returning selected AMReX objects through pyAMReX's registered types.
5. WarpX hides extension symbols and excludes symbols from statically linked dependencies to reduce cross-module symbol collisions. See its [Python module target configuration](https://github.com/BLAST-WarpX/warpx/blob/2e5f0a26691c5378f75bd337a34d2b5bd5653d48/CMakeLists.txt).
6. Its Python field wrapper re-fetches the underlying `MultiFab` from the C++ registry on every access because regridding can recreate it. See [`fields.py`](https://github.com/BLAST-WarpX/warpx/blob/2e5f0a26691c5378f75bd337a34d2b5bd5653d48/Python/pywarpx/fields.py).

This is strong evidence that pyAMReX and custom application bindings can form a production CPU/CUDA/HIP Python interface without moving the solver core into Python. It also identifies constraints that are architectural rather than incidental: one pybind11 type universe, coordinated builds, dimension-aware loading, carefully defined initialization ownership, and explicit state-lifetime rules.

pyAMReX still does not eliminate the need for Quokka bindings. It has no knowledge of `Runtime`, `PhysicsConfig`, `ProblemPlan`, `StateSchema`, `OperatorGraph`, or `Simulation`. Using its `AmrCore` trampoline as the solver foundation would place low-level AMR allocation and regrid callbacks in Python, contrary to this decision's C++ ownership model. Quokka therefore needs a custom extension, but it should use pybind11 so it can reuse pyAMReX types directly.

| Criterion | pyAMReX alone | Custom nanobind module | Layered pybind11 plus pyAMReX |
|---|---|---|---|
| Bind Quokka runtime and solver concepts | No | Yes | Yes |
| Preserve C++ ownership of AMR and scheduling | Possible, but no application control API | Yes | Yes |
| Reuse AMReX `MultiFab` and particle bindings | Yes | No direct type reuse | Yes |
| CPU/CUDA array interoperability | Available | Must be reimplemented | Available |
| MPI/GPU build matrix | Available, if built compatibly | Quokka must implement the data layer | Available, if built as one coordinated bundle |
| Stable public control API | Too low-level on its own | Narrow | Narrow Quokka API with an opt-in AMReX data plane |
| Type-registry complexity | One pybind11 registry | Separate nanobind registry | One shared pybind11 registry |
| Regrid-safe state lifetime | Application policy still required | Application policy still required | Application policy still required; WarpX supplies a proven re-fetch pattern |
| Packaging complexity | Dimension/backend-specific | One custom extension, but duplicated bindings | Coordinated Quokka and pyAMReX modules per execution bundle |

The resulting policy is:

1. `_quokka_core` is implemented with pybind11 and primarily binds Quokka-owned types.
2. Enabling Quokka's Python bindings also builds or requires a specifically compatible pyAMReX. Both modules must use the exact same AMReX source, compiler and C++ ABI, precision, MPI mode, GPU backend, dimension, and pybind11 internals version.
3. The build writes those compatibility properties into a machine-readable bundle fingerprint. The package loader checks the fingerprint, imports the matching `amrex.spaceXd` module before `_quokka_core`, and rejects incompatible or multiple-dimension loads with an actionable diagnostic.
4. Importing either module does not initialize AMReX. `quokka.Runtime` remains the sole initialization/finalization owner for a Quokka process.
5. `Simulation.run()` and `step()` release the GIL with `py::call_guard<py::gil_scoped_release>()` or an equivalent explicit guard.
6. Recoverable Quokka errors are translated to a small hierarchy of Python exceptions.
7. The stable control interface uses Quokka concepts. Selected advanced field and particle accessors may return pyAMReX types, but raw `MFIter` and per-cell Python callbacks are not part of the solver control path.
8. Initial state inspection supports copies and reductions as the safe default. A field accessor re-fetches the current C++ `MultiFab` by field identifier and level rather than caching a wrapper across regrid.
9. An exported zero-copy NumPy, CuPy, or future DLPack array additionally requires a Quokka-owned paused-state lease defining device, stream, MPI locality, mutability, and regrid behavior. Re-fetching the `MultiFab` does not make a previously exported pointer safe.
10. Quokka follows WarpX's hidden-symbol discipline for its extension and avoids statically embedding conflicting copies of AMReX or other shared dependencies in separately loaded modules.
11. Quokka does not add application-specific pyAMReX particle-layout specializations. It returns pyAMReX's generic polymorphic PureSoA type during the migration bridge and the generic runtime-only type when that binding becomes available. Different Quokka schemas must have the same Python container type.

An illustrative Python problem is:

```python
import quokka

run = quokka.RunConfig.from_toml("inputs/HydroShocktube.toml")

physics = quokka.Physics(
    hydro=quokka.Hydro(
        eos=quokka.eos.ideal_gas(
            gamma=1.4,
            mean_molecular_weight=quokka.constants.m_u,
        ),
        passive_scalars=[],
    )
)

problem = quokka.Problem(
    initial_conditions=quokka.initializers.shock_tube(
        axis="x",
        interface=2.0,
        left=quokka.PrimitiveState(density=10.0, pressure=100.0),
        right=quokka.PrimitiveState(density=1.0, pressure=1.0),
    ),
    boundaries=quokka.boundaries.constant_x(
        low=quokka.PrimitiveState(density=10.0, pressure=100.0),
        high=quokka.PrimitiveState(density=1.0, pressure=1.0),
    ),
    refinement=quokka.refinement.relative_gradient(
        field="gasDensity", threshold=0.1, minimum=0.01
    ),
)

with quokka.Runtime() as runtime:
    sim = quokka.Simulation(runtime, run, physics, problem)
    sim.initialize()
    sim.run()
```

A multigroup configuration changes data, not types:

```python
physics = quokka.Physics(
    hydro=quokka.Hydro(eos=quokka.eos.ideal_gas(gamma=5.0 / 3.0)),
    radiation=quokka.Radiation(
        group_boundaries=[1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19],
        opacity=quokka.opacity.piecewise_power_law(table="opacity.h5"),
    ),
)
```

No CMake configure or C++ build occurs when the list length changes.

### GIL, MPI, and errors

- `Simulation.run()` and `step()` release the GIL during C++ execution.
- The GIL is reacquired only for an explicit Python lifecycle or patch callback.
- Observation callbacks default to the I/O rank.
- Collective callbacks are explicitly marked and entered by every rank.
- Callback failures are coordinated before any rank enters the next collective operation.
- Recoverable configuration and registry errors throw binding-safe exceptions instead of calling `amrex::Abort`.
- Process abort remains acceptable for unrecoverable distributed corruption.

### State inspection

The first binding should expose:

- state schema and metadata;
- scalar diagnostics and reductions;
- one-dimensional extracts;
- explicit host copies of selected fields/patches; and
- controlled checkpoint/plotfile requests.

Zero-copy NumPy, CuPy, or DLPack views should wait until the interface defines:

- valid versus ghost cells;
- patch and level iteration;
- host versus device memory;
- GPU stream ownership;
- regrid invalidation;
- MPI locality; and
- read-only versus mutable access.

An exported raw array cannot perform a generation check on every later Python access. The zero-copy design must therefore use a lease acquired only while the simulation is paused. While any lease is active, operations that can reallocate or invalidate storage—at minimum regrid, level removal, simulation destruction, and possibly stepping—must fail or wait. Releasing the last lease permits those operations again. An explicit copy remains the default when that restriction is undesirable.

## Feasibility by subsystem

| Subsystem | Feasible without per-problem compilation? | Main work |
|---|---|---|
| AMR state allocation | Yes | Build and freeze runtime field schema. |
| Hydro enablement and parameters | Yes | Remove `problem_t`, use field indices and runtime config. |
| Passive/mass scalars | Yes | Replace fixed arrays with ranges/spans and runtime loops. |
| MHD enablement | Yes | Runtime operator and optional face-centered field allocation. |
| Radiation group count | Yes, substantial refactor | Replace fixed arrays/Jacobians with spans and workspace. |
| Dust group count | Yes | Replace fixed group arrays with spans/workspace. |
| Existing EOS selection | Yes | Compiled model registry and launch/device dispatch. |
| Existing opacity selection | Yes | Compiled model registry and runtime tables. |
| New EOS or opacity | Compilation expected | Add model implementation and registration. |
| Existing chemistry network | Yes within its bundle | Runtime selection/configuration if linked. |
| New chemistry network | Compilation expected | Generate/build network module or bundle. |
| Chemistry integration | Yes | Use the shared vendored Rosenbrock operator with runtime tolerances. |
| Remove Microphysics | Yes; redesign prerequisite | Port EOS plumbing, network contracts, Rosenbrock behavior, and build integration into Quokka-owned modules. |
| Particle components | Yes | Use a `ParticleStore`, one generic polymorphic PureSoA container type, and runtime schemas; move to AMReX's runtime-only container behind the same interface. |
| Existing particle processes | Yes | Runtime operator selection. |
| New particle dynamics | Compilation expected | Add physics operator. |
| Initial conditions/boundaries | Yes | Expressions, built-in launchers, and patch fallback. |
| Refinement/derived fields | Yes | Generic stencils/expressions and patch fallback. |
| Python orchestration | Yes | Bind non-template interfaces and enforce lifetimes. |

There is no architectural blocker. The difficult portions are performance engineering and migration volume, especially multigroup radiation and the current templated particle call sites.

## Alternatives considered

### Bind every current problem instantiation

This proves Python/AMReX lifetime handling but retains per-problem compilation and does not meet the compilation policy.

### Replace specializations with one virtual problem class

This removes syntax-level specialization but preserves a broad hook interface and cannot safely implement arbitrary GPU work in Python. It is an incremental adapter, not a target architecture.

### Precompile a catalog of solver shapes

This removes named problem types but still requires compilation for a previously unseen group count, scalar count, particle layout, or physics combination. It does not meet the stronger requirement.

It can be a temporary bridge while subsystems become dynamic, but the generic runtime path must remain available and must not fail because an exact shape was not pre-instantiated.

### Fully runtime state plus compiled operator/model registries

This is the recommended target.

Advantages:

- exactly matches the compilation policy;
- gives C++ and Python one stable interface;
- decouples state schema from problem identity;
- makes existing physics composable at runtime;
- avoids a solver-shape combinatorial explosion; and
- keeps new executable physics as an explicit compiled extension point.

Costs:

- larger refactor than finite explicit instantiation;
- runtime-loop and workspace redesign for multigroup radiation/dust;
- operator-order validation becomes first-class infrastructure;
- performance fast paths need careful host dispatch; and
- existing problem specializations need systematic migration.

### Use pyAMReX without custom Quokka bindings

This would expose AMReX building blocks but not the runtime solver, physics graph, problem plan, lifecycle invariants, or Quokka diagnostics. Building the solver by subclassing pyAMReX `AmrCore` in Python would move too much AMR control into Python and would not provide the intended production architecture.

### Custom nanobind bindings with protocol-only pyAMReX interoperability

This would produce a narrow Quokka interface, but nanobind and pyAMReX would maintain different C++ type registries. Direct reuse of pyAMReX's `MultiFab` and particle bindings would be lost; Quokka would either duplicate them or restrict interoperability to copies, DLPack, or versioned capsules. WarpX shows that this duplication is unnecessary. This alternative is rejected unless Quokka later decides that raw AMReX interoperability has no supported use case.

### Layer custom pybind11 bindings with pyAMReX

This is the recommended binding design. The stable `_quokka_core` module binds Quokka's deep interfaces, while compatible pyAMReX modules provide the low-level AMReX data plane. The modules share one pybind11 type registry and one exact AMReX build. High-level users need not manipulate AMReX objects, but advanced workflows do not require Quokka to recreate their bindings.

## Proposed source and build organization

```text
src/runtime/
  Runtime.hpp/.cpp
  RunConfig.hpp/.cpp
  PhysicsConfig.hpp/.cpp
  StateSchema.hpp/.cpp
  Simulation.hpp/.cpp

src/operators/
  OperatorGraph.hpp/.cpp
  hydro/
  radiation/
  mhd/
  dust/
  gravity/
  conduction/
  chemistry/
  particles/

src/particles/
  ParticleSchema.hpp/.cpp
  ParticleStore.hpp/.cpp
  ParticleLaunchView.hpp
  AmrexParticleStorage.hpp/.cpp

src/models/
  ModelRegistry.hpp/.cpp
  eos/
  opacity/
  chemistry/

extern/rosenbrock/
  LICENSE
  UPSTREAM.md
  include/
  patches/

src/problem/
  ProblemPlan.hpp/.cpp
  ProblemLifecycle.hpp
  expressions/
  stencils/
  initializers/
  boundaries/
  refinement/
  diagnostics/

src/python/
  module.cpp
  BindingErrors.hpp/.cpp
  PatchView.hpp/.cpp
  bind_runtime.cpp
  bind_config.cpp
  bind_problem.cpp
  bind_simulation.cpp

python/quokka/
  __init__.py
  eos.py
  opacity.py
  initializers.py
  boundaries.py
  refinement.py
  diagnostics.py
```

Build outputs:

```text
quokka_runtime          non-template AMR engine and configuration
quokka_operators        compiled existing physics operators
quokka_models           compiled EOS/opacity catalog for the bundle
quokka_rosenbrock       network-independent vendored ODE integration module
quokka_chemistry_*      generated network module or bundle-specific library
_quokka_core            thin custom pybind11 module over Quokka interfaces
pyAMReX spaceXd module  compatible low-level AMReX data bindings
quokka-run              generic C++ driver using the same interfaces
```

The current `QUOKKA_PYTHON` option embeds Python/NumPy for plotting. Binding support should use a separate option such as `QUOKKA_ENABLE_PYTHON_BINDINGS` so embedded plotting can be retired independently.

Initially, operator and model registries may be linked statically into the execution bundle. A versioned shared-plugin interface can follow after CPU/CUDA/HIP linking and ABI behavior are proven. The compilation policy does not require dynamic loading; it requires that only new executable physics triggers a rebuild.

## Detailed migration plan

### Stage 0: approve the compilation rule and establish baselines

- Write an ADR recording the “configuration is data; new executable physics is compiled” rule.
- Generate a machine-readable inventory of traits, group counts, scalar counts, particle layouts, specializations, and operator combinations.
- Record build time, aggregate binary size, CPU/GPU time-to-solution, GPU register counts, plotfile schemas, and checkpoint metadata.
- Select parity problems for hydro, MHD, grey and multigroup radiation, dust, particles, gravity, tabulated EOS, and chemistry.
- Add a CI concept test that builds once and runs multiple configurations without rebuilding.

Exit criterion: maintainers agree that group/scalar counts and existing operator combinations are runtime requirements, not optional future work.

### Stage 1: introduce runtime schema and field handles

- Implement `StateSchemaBuilder`, `StateSchema`, `FieldId`, field ranges, centering, units, and schema serialization.
- Add an adapter that constructs a runtime schema from current `Physics_Traits<problem_t>` and `Physics_Indices<problem_t>`.
- Resolve semantic field names to integer launch data before kernels execute.
- Make plotfile/checkpoint naming consume the runtime schema.
- Add schema equality/restart tests.

Existing solver templates remain in this stage; the schema is initially a faithful mirror.

Exit criterion: all state allocation and I/O metadata can be described by a runtime schema even while legacy kernels still use static indices internally.

### Stage 2: add `ProblemPlan` and a no-compile problem path

- Implement expression initialization with primitive-to-conserved conversion.
- Implement constant/expression boundaries.
- Implement generic region, threshold, and gradient taggers.
- Implement Python patch-copy fallback for initialization, refinement, and derived fields.
- Add lifecycle contexts and binding-safe validation.
- Run `HydroShocktube` through a generic C++ driver while retaining the legacy solver internally.

Exit criterion: changing shock states, interface position, boundary values, and refinement thresholds requires no C++ source or build change.

### Stage 3: make hydro and state allocation non-templated

- Create a non-template AMR simulation core.
- Convert hydro field indices to runtime `HydroFields`.
- Convert passive and mass scalar paths to runtime ranges.
- Introduce the EOS registry and move ideal/tabulated EOS parameters out of traits.
- Implement Quokka-owned ideal-gas EOS initialization and parameter storage so even non-chemistry builds no longer rely on Microphysics EOS headers or globals.
- Dispatch EOS-specific launches outside cell loops.
- Convert hydro boundary/fixup/interpolation paths to runtime fields.
- Preserve optimized direction templates internally.

Exit criterion: one built hydro operator runs arbitrary passive-scalar counts and any compiled EOS model without a problem template.

This is the first useful Python-binding milestone.

### Stage 4: vendor Rosenbrock and remove Microphysics

- Freeze reference behavior before extraction: RHS/Jacobian values, accepted/rejected steps, error weights, endpoint cleanup, failure/retry results, and conserved mass/charge/energy for representative chemistry problems.
- Define the Quokka-owned `ChemistryNetwork` and immutable device-data contracts, including per-variable error-control participation and state validity rules.
- Vendor the required Rosenbrock implementation with its license, pinned upstream provenance, local-patch record, and update instructions.
- Port tableau evaluation, adaptivity, rejection/retry, tolerance scaling, endpoint cleanup, and small dense linear solves; retain the specialized six-equation solve only behind the generic linear-solver interface.
- Port the current chemistry and photoionization networks behind the new interface, preserving species ordering, thermodynamic coupling, and passive radiation-flux semantics.
- Replace Microphysics `burn_t`, EOS types, generated `actual_*` includes, and global/external parameter objects with Quokka-owned state, model data, and typed runtime configuration.
- Route all thermodynamic conversions through the EOS registry introduced in Stage 3.
- Replace `setup_target_for_microphysics_compilation` and related target-wide definitions with ordinary Quokka network-module targets linked to `quokka_rosenbrock`.
- Remove `add_subdirectory(extern/Microphysics)` and all remaining production includes/linkage after parity is established.
- Validate the same network modules on CPU and applicable CUDA/HIP backends.

Exit criterion: representative primordial-chemistry and photoionization problems run with the vendored Rosenbrock path, and no production Quokka target depends on Microphysics source, generated headers, CMake helpers, parameters, or types.

This stage is a prerequisite for claiming that a new network is the only chemistry-specific compilation unit. It should be completed before the final non-template solver and Python interface are declared stable, even if some independent operator migrations proceed in parallel.

### Stage 5: runtime MHD, gravity, conduction, cooling, and source ordering

- Extract existing physics paths into runtime operator descriptors.
- Allocate face-centered MHD state only when requested by the runtime schema.
- Build and validate the operator phase graph.
- Replace `if constexpr` scheduling with immutable runtime phase lists.
- Keep operator implementation templates private where useful.

Exit criterion: enabling/disabling any existing operator in this stage changes only configuration.

### Stage 6: runtime multigroup radiation

- Replace `nGroups` type parameters in state layout with runtime ranges.
- Replace fixed group arrays with device spans.
- Design `RadiationWorkspace` for residuals, opacities, and nonlinear solve scratch.
- Refactor the Jacobian solve to runtime group counts, exploiting structured algebra.
- Introduce the opacity registry and runtime group-boundary tables.
- Keep an optional `n_groups == 1` optimized launch selected at runtime.
- Compare generic and optimized paths for representative group counts.

Exit criterion: one built radiation operator runs the repository's current group-count matrix plus previously unused counts without recompilation.

### Stage 7: runtime dust groups

- Replace fixed dust arrays with ranges and workspace.
- Move drag/charge parameters into compiled model data selected at runtime.
- Treat a new dust interaction law as a new compiled operator/model.

Exit criterion: dust group count and existing dust options are runtime data.

### Stage 8: runtime particle schemas and processes

- Inventory and remove direct assumptions about `ArrayOfStructs`, `ParticleType::NReal`, `ParticleType::NInt`, `p.rdata()`, and `p.idata()` across particle creation, deposition, accretion, destruction, motion, I/O, and restart.
- Add the deep `ParticleStore` module and freeze named `ParticleSchema` values before arena selection or particle allocation.
- Prototype the already-bound polymorphic PureSoA bridge with positions as its only compile-time fields and all Quokka fields as runtime components.
- Keep separate logical populations where their lifecycle requires it, but instantiate the same concrete container type for every population.
- Replace `Particle_Traits<problem_t>` switches with runtime population and process configuration.
- Build operator-specific `ParticleLaunchView` values on the host with direct component pointers and counts; do not perform name lookup or repeated component-offset arithmetic inside kernels.
- Update creation, redistribution, deposition, checkpoint/restart, and plotfile paths to consume runtime schema metadata.
- Convert radiation luminosity group fields and stellar-model extras to runtime group-sized components.
- Evaluate `amrex::ParticleContainerRTSoA<>` feature parity for Quokka on CPU, CUDA, and HIP. Switch the storage adapter when required functionality and pyAMReX exposure are ready; do not expose the bridge type in Quokka's public interface.
- Upstream a generic pyAMReX runtime-only particle binding if it has not landed, rather than adding Quokka layout specializations.

Exit criterion: existing particle families, group counts, stellar-model fields, and component counts use the same C++ and Python container types without compilation; a new evolution law remains a compiled operator. Redistribution, deposition, I/O, restart, and representative hot kernels meet correctness and performance gates on the supported backends.

### Stage 9: integrate chemistry modules with the runtime schema

- Register the Stage 4 network modules with the operator/model registries.
- Make network parameters, tolerances, and enablement runtime values.
- Map compiled species metadata into the runtime state and particle schemas without exposing network-specific C++ types.
- Decide whether bundles contain one network or multiple linked network modules.
- Validate operator ordering for chemistry, radiation coupling, cooling, and Strang-split sources.

Exit criterion: Python selects and configures an available chemistry network through the uniform simulation interface, while adding a new RHS/Jacobian network remains an explicit build operation against the shared Rosenbrock integrator.

### Stage 10: complete Python bindings

- Pin pyAMReX and pybind11 versions and force pyAMReX to use the execution bundle's exact AMReX source and configuration.
- Add a custom pybind11 `_quokka_core` target linked directly to the execution bundle.
- Generate a bundle fingerprint covering the AMReX source revision and options, compiler/C++ ABI, precision, MPI mode, GPU backend, dimension, and pybind11 internals version.
- Add a dimension-aware loader that validates the fingerprint, imports the matching `amrex.spaceXd` module before `_quokka_core`, and rejects incompatible or multiple-dimension loads.
- Apply hidden symbol visibility and dependency-linking rules that prevent separately loaded Python modules from embedding conflicting AMReX or third-party symbols.
- Bind `Runtime`, configurations, model builders, `ProblemPlan`, schema, simulation control, and scalar diagnostics as Quokka-owned types.
- Release the GIL around C++ execution with a pybind11 call guard.
- Translate recoverable configuration, registry, schema, and lifecycle failures to typed Python exceptions.
- Test callback failure coordination under MPI.
- Start with explicit state copies and reductions.
- Add an advanced field accessor that re-fetches the current `MultiFab` from the C++ field registry and returns the already registered pyAMReX type.
- Add a population accessor that returns the generic pyAMReX polymorphic PureSoA container during the bridge. Require two different `ParticleSchema` values to produce the same Python type.
- When pyAMReX binds `amrex::ParticleContainerRTSoA`, update only the storage adapter and population accessor; keep the Quokka Python interface unchanged.
- Prototype the paused-state lease/epoch protocol before exporting zero-copy arrays from a pyAMReX `MultiFab`; reject regrid or destruction while a lease is active.
- Extend that lease protocol to particle arrays: redistribution, sorting, creation, deletion, tile growth, arena migration, and destruction can invalidate exported pointers and must be blocked or copied.
- Add packaging/import tests for the exact MPI/GPU toolchain.

Exit criterion: representative hydro, multigroup radiation, and particle problems run through `mpiexec ... python` with no problem executable; advanced field access returns the expected pyAMReX type without a copied or duplicate AMReX binding; and distinct runtime particle schemas return the same generic pyAMReX container type.

### Stage 11: migrate and remove legacy problem specializations

Migrate problems by complexity:

1. simple declarative tests such as shock tubes and waves;
2. table-driven problems such as `RadhydroShell`;
3. stateful multiphysics problems such as `DiskGalaxy`; and
4. chemistry/network problems.

For each migration:

- move scenario composition to Python or a data plan;
- move reusable numerical behavior into an existing or new compiled model/operator;
- move reference checks into reusable validators;
- run legacy/runtime parity; and
- remove the named executable and specializations only after parity.

Finally:

- deprecate `quokka_add_problem` for ordinary problems;
- update `developing_problem_generators.md`;
- remove empty problem tags and `SimulationData<Problem>` specializations; and
- stop including templated solver implementations in user code.

## Mapping current constructs to the target

| Current construct | Target owner |
|---|---|
| Empty `struct Problem` | Removed |
| `Physics_Traits<Problem>` enabled flags | Runtime `PhysicsConfig` and operator graph |
| `Physics_Traits` group/scalar counts | Runtime state schema/ranges |
| `Physics_Indices<Problem>` | Runtime `StateSchema` and resolved field structs |
| Scalar `EOS_Traits` values | Runtime EOS model data |
| EOS backend type | Compiled EOS registry selected at runtime |
| Microphysics EOS and global initialization | Quokka-owned EOS model data and registry initialization |
| Microphysics `burn_t`, generated headers, and global network parameters | Quokka `ChemistryNetwork`, typed runtime parameters, and field mapping |
| Microphysics Rosenbrock/VODE integration | Vendored network-independent Rosenbrock operator |
| `RadSystem_Traits` group boundaries/parameters | Runtime radiation/opacity data |
| `Particle_Traits` and component aliases | Runtime particle schema and process graph |
| `SimulationData<Problem>` | Owned state in lifecycle/model/launcher modules |
| `setInitialConditionsOnGrid` | Expression or registered initializer plan |
| `setInitialConditionsOnGridFaceVars` | Face-field initializer plan |
| custom boundary specialization | Boundary expression/launcher |
| `refineGrid` | Generic stencil/expression tagger |
| `addStrangSplitSources` | Ordered source/operator phase |
| opacity specialization | Runtime-selected compiled opacity model |
| derived-variable specialization | Expression/stencil/registered derived field |
| `ComputeStatistics` | Diagnostic/reduction plan |
| before/after timestep hooks | Coarse lifecycle callback or operator phase |
| reference solution and return code | Reusable Python/C++ validator |
| `problem_main()` | Python script or generic C++ driver |
| per-problem `add_executable` | Data-driven CTest/Python invocation |

## Testing and acceptance criteria

### The no-rebuild test

CI should build an execution bundle once, record build outputs and timestamps, then run a configuration matrix that varies:

- zero, one, and several passive scalars;
- several radiation group counts, including a count not used by a legacy problem;
- several dust group counts;
- MHD/gravity/conduction enablement;
- existing EOS and opacity selections;
- particle process combinations and runtime component counts; and
- multiple Python-defined initial/boundary/refinement plans.

No compilation or relink may occur between runs.

### Interface tests

- schema duplicate fields, missing dependencies, invalid centering, and invalid group ranges fail before allocation;
- operator ordering cycles and missing required operators fail clearly;
- model parameters and compatibility are validated;
- schema mutation after freeze fails;
- particle schemas reject duplicate/reserved names and mutation after arena selection or allocation;
- state views are invalidated safely on regrid or destruction; and
- Python objects cannot outlive `Runtime`.

Binding-specific acceptance tests additionally require:

- importing pyAMReX and `_quokka_core` does not initialize AMReX;
- `Runtime` initializes and finalizes exactly once, including exception paths;
- `run()` and `step()` release the GIL and reacquire it only for declared callbacks;
- MPI callback exceptions are reported coherently without leaving another rank in a collective;
- the extension imports and executes a smoke problem for CPU and applicable CUDA/HIP bundles;
- the matching `amrex.spaceXd` module is loaded first and Quokka-returned AMReX objects have the expected pyAMReX Python type;
- incompatible AMReX, pybind11, dimension, precision, MPI, or GPU-backend combinations fail at import with an actionable diagnostic;
- the packaged modules do not load duplicate AMReX or conflicting statically linked dependency symbols;
- state copies have explicit valid/ghost/component ordering;
- field wrappers re-fetch storage after regrid;
- distinct particle schemas return the same generic pyAMReX particle-container type;
- changing a particle arena after allocation fails, while an explicit copy/migration operation preserves schema and values;
- particle-array leases block redistribution, sorting, creation/deletion, tile reallocation, and destruction; and
- a zero-copy prototype, if enabled, prevents regrid, storage replacement, or destruction until its lease ends.

### Numerical parity

For each migrated problem:

- compare conserved sums, error norms, step counts, AMR grids, and diagnostics;
- compare checkpoint restart continuation;
- use bitwise checks when the launch implementation is unchanged;
- otherwise retain existing scientific tolerances; and
- test CPU plus applicable CUDA/HIP backends.

Chemistry extraction additionally requires differential tests against the frozen Microphysics reference for:

- RHS and analytic Jacobian evaluation;
- one Rosenbrock step and full one-zone trajectories;
- error weighting, accepted/rejected steps, retry, and terminal failure;
- species positivity, mass and charge closure, and gas/radiation energy exchange;
- passive integrated variables that do not participate in error control; and
- the current six-equation photoionization solve on CPU and GPU backends.

### Performance

Measure:

- clean and incremental build time;
- aggregate library/binary size;
- runtime dispatch overhead;
- passive-scalar scaling;
- radiation/dust group-count scaling;
- radiation workspace memory;
- GPU registers and occupancy;
- per-cell instruction counts and address arithmetic for runtime versus static field indices;
- operator graph overhead;
- Python callback disabled/enabled overhead; and
- end-to-end time per cell update.

A proposed initial gate is less than a 2% end-to-end regression for configurations whose kernel mathematics is unchanged. The dynamic multigroup rewrite should define separate memory and performance gates after a prototype, because removing fixed local arrays changes the implementation materially.

## Risks and mitigations

### Runtime multigroup scratch

**Risk:** dynamic per-cell arrays cause heap allocation, local-memory spills, or excessive global workspace.

**Mitigation:** use spans into state, structured algebra, preallocated workspace, streamed group loops, and optional common-count fast paths. Prototype this before broad radiation migration.

### Model cross-product compilation

**Risk:** host-specializing every operator for every EOS and opacity recreates template combinatorics.

**Mitigation:** choose launch dispatch only where it materially improves performance; use a compact uniform device model tag elsewhere. Compile by model/operator, never by problem or group count.

### Runtime index arithmetic

**Risk:** replacing static component constants with runtime fields accidentally adds repeated offset calculations, metadata loads, abstraction overhead, or enough live indices to increase GPU register pressure in every cell update.

**Mitigation:** resolve names and ranges before launch, precompute base offsets and strides in compact operator-specific index packs, keep index accessors trivially inlineable, and inspect instruction counts, registers, occupancy, and end-to-end kernel timings against the static baseline. Make this comparison a migration gate for each hot kernel rather than assuming that equivalent source expressions compile equivalently.

### Rosenbrock semantic drift

**Risk:** extracting only the obvious stage equations changes tolerance weighting, step rejection, cleanup, retry, passive-variable handling, or conservation behavior. A solver that compiles and passes a simple trajectory can still change production chemistry.

**Mitigation:** freeze Microphysics as a temporary differential reference, test each semantic layer independently, and require one-zone plus full-problem parity before removing it. Treat variable participation in error control and validity checks as network metadata rather than an integrator assumption.

### Vendored integrator ownership

**Risk:** copying code without provenance turns a dependency removal into an unmaintainable fork.

**Mitigation:** vendor only the narrow Rosenbrock and linear-algebra implementation, preserve licensing and upstream revision metadata, isolate Quokka adapters, record every local patch, and provide a repeatable upstream comparison/update procedure.

### Operator-order mistakes

**Risk:** moving `if constexpr` paths into a runtime graph permits invalid combinations or orderings.

**Mitigation:** operators declare fields, prerequisites, conflicts, and ordering constraints. Freeze and print the graph before allocation; cover combinations in tests.

### Python/MPI failure semantics

**Risk:** one rank raises while another enters a collective and hangs.

**Mitigation:** restrict callbacks to explicit synchronization phases, declare rank scope, coordinate failure flags, and test callback exceptions under MPI.

### AMReX lifetime

**Risk:** import-time initialization or lingering Python views violate AMReX lifetime rules.

**Mitigation:** explicit `Runtime` ownership, generation-checked views, one active runtime initially, and destruction-order tests.

### GPU views

**Risk:** a zero-copy Python view is invalid after regrid or uses the wrong stream.

**Mitigation:** start with copies/reductions. Add zero-copy protocols only with explicit stream and locality metadata plus a paused-state lease that prevents regrid, level removal, storage replacement, and destruction while the exported pointer exists. A generation counter alone cannot protect an already exported raw array.

### Cross-module binding ABI and symbol collisions

**Risk:** pyAMReX and `_quokka_core` load incompatible pybind11 registries, bind the same AMReX type twice, or embed conflicting copies of AMReX and other statically linked dependencies. The result can range from import-order errors to incorrect ownership or process crashes.

**Mitigation:** follow the WarpX superbuild pattern: use one exact AMReX source and configuration, one compatible pybind11 version, import pyAMReX first, return rather than rebind its AMReX types, hide module symbols, exclude archive symbols where required, and test the complete CPU/CUDA/HIP and MPI packaging matrix.

### Particle runtime-component performance

**Risk:** moving all particle fields to runtime PureSoA storage adds component-index arithmetic, changes memory access, or increases the cost of packing, redistribution, deposition, and checkpoint operations.

**Mitigation:** resolve component names and `component * capacity` offsets on the host, pass direct pointers in `ParticleLaunchView`, and use AMReX's runtime communication flags. Compare the polymorphic PureSoA bridge and `amrex::ParticleContainerRTSoA` against current Quokka kernels on CPU, CUDA, and HIP, measuring kernel time, instructions, registers, occupancy, redistribution, deposition, and I/O. Treat the upstream HiPACE++ result as feasibility evidence, not a substitute for Quokka benchmarks.

### Upstream particle-interface transition

**Risk:** Quokka exposes an AMReX or pyAMReX bridge type that is renamed or superseded while the runtime-only particle roadmap is still landing. The current AMReX runtime-only container is available, but its generic pyAMReX binding is not yet present in the inspected revision.

**Mitigation:** keep `ParticleStore` as the deep module and the only Quokka-owned interface to storage. Pin the exact AMReX/pyAMReX revisions, maintain bridge and target adapters behind the same seam, upstream the one generic binding rather than Quokka-specific shapes, and require feature-parity tests before switching adapters.

### Restart compatibility

**Risk:** runtime schema assembly changes component ordering.

**Mitigation:** initially reproduce existing ordering, serialize the complete schema, and require explicit schema compatibility checks on restart.

### Runtime configuration errors currently abort

**Risk:** `amrex::Abort` terminates Python for recoverable mistakes.

**Mitigation:** move validation ahead of distributed execution and use structured errors in registries/configuration. Reserve aborts for unrecoverable runtime corruption.

## Recommended first implementation slices

The complete design is large. Three vertical slices can prove it without pretending a finite shape catalog is the endpoint.

### Slice 1: runtime hydro problem

1. Add runtime schema and field handles.
2. Add `ProblemPlan` expressions, shock-tube launcher, boundaries, and tagger.
3. Add non-template hydro state allocation and scalar ranges.
4. Add ideal-gas EOS runtime model.
5. Run arbitrary passive-scalar counts through generic C++ and Python drivers.

This proves the public seam and no-compile problem authoring.

### Slice 2: dynamic multigroup prototype

1. Isolate a representative matter-radiation update.
2. Replace fixed group arrays with spans/workspace.
3. Run group counts 1, 2, 5, and a previously unused count from one binary.
4. Compare numerical results, workspace memory, registers, and runtime.
5. Decide structured solver and fast-path policy.

This resolves the largest technical risk before rewriting the whole radiation module.

### Slice 3: runtime particles

1. Add `ParticleStore`, `ParticleSchema`, and the polymorphic PureSoA bridge with only positions fixed by dimension.
2. Configure two logical populations with different CIC, radiation-luminosity, and integer fields from runtime schemas while using the same concrete C++ and Python types.
3. Convert one representative creation, drift/kick, deposition, redistribution, I/O, and restart path away from AoS and compile-time component counts.
4. Build direct-pointer `ParticleLaunchView` values on the host and compare kernel/index arithmetic with the current static layout.
5. Run different group/component counts from one binary and validate CPU plus one GPU backend.
6. Prototype the same slice with `amrex::ParticleContainerRTSoA`; record missing AMReX/pyAMReX capabilities and upstream the generic binding if needed.

This proves the remaining state-shape concern using facilities already present in the vendored AMReX.

### Slice 4: Rosenbrock extraction

1. Freeze single-zone primordial-chemistry and six-equation photoionization reference cases against the current Microphysics revision.
2. Define the smallest network RHS/Jacobian, field-mapping, validity, and error-control contract that supports both cases.
3. Vendor and adapt Rosenbrock without exposing a Microphysics state type.
4. Differentially compare stages, error estimates, rejection/retry, final state, and conservation on CPU and one GPU backend.
5. Switch those problems to the new path, then remove their Microphysics build setup.

This proves that compilation follows the chemistry network rather than the problem, and it retires the incompletely templated dependency before the binding interface hardens around it.

## Decisions required before implementation

1. What are the stable phase roles and ordering rules in the operator graph?
2. Which EOS operations form the compiled model interface?
3. Which opacity operations belong in model modules versus radiation operators?
4. What dynamic multigroup workspace and structured solve should replace fixed local arrays?
5. Which logical particle populations require separate instances of the same generic storage type, and which can share one population distinguished by a runtime `kind` field?
6. Is `amrex::ParticleContainerRTSoA` feature-complete for Quokka's AMR, deposition, creation, restart, and I/O paths, and what must be upstreamed before replacing the PureSoA bridge?
7. What exact Rosenbrock source revision and license will be vendored, and what upstream-update policy will be maintained?
8. Which network metadata controls error weighting, validity, conservation, and passive integrated variables?
9. Is one chemistry network per execution bundle acceptable initially?
10. Does the generic small dense solve need only a six-equation optimized path initially, or additional fixed-size kernels?
11. Which expression and stencil operations are required for the universal no-compile problem path?
12. What state-copy interface is sufficient before zero-copy Python views, and which mesh and particle operations must a future view lease block?
13. Which recoverable AMReX error paths must be converted for Python?

## Expected outcome

After migration, changes fall into two categories.

### Configuration and problem definition: no compilation

Write or modify a Python script to:

- choose existing physics operators;
- choose an existing EOS/opacity/network already in the bundle;
- set group/scalar/particle schemas;
- specify initial and boundary data;
- configure refinement and diagnostics; and
- drive the run.

### New executable physics: compilation

Add a focused compiled module for:

- a new EOS;
- a new opacity;
- a new chemistry network;
- a new physics/particle operator; or
- an optional optimized launcher.

For chemistry, a new network implements the Quokka-owned network contract and links to the already vendored Rosenbrock implementation; it does not import Microphysics or create a new solver type. The simulation type, state layout, and Python interface do not change. That is the key architectural result: compilation tracks new executable physics, not new problems or new runtime dimensions.
