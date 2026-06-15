# Modular Stellar-Evolution Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compile-time-selectable stellar-evolution model framework for Quokka `Star` particles, with a stateless analytic toy model as the default, plus a new test problem (`ParticleStarEvolution`) that validates the toy model's radius and luminosity against closed-form solutions.

**Architecture:** A `StellarModel_Traits<problem_t>` trait selects a model struct (default `ToyStellarModel`) exposing GPU device functions for `R(M)`, `L_star(M)`, and accretion luminosity. A dispatcher `StellarUpdate::updateStellarProperties` (reached through the existing `ParticlePropertyUpdateTraits<ParticleType::Star>` mechanism) reads particle fields, calls the model, and writes back `radius` and `lum`. The `Star` particle type, its container, the accretion `mdot` wiring, and the update dispatch are all (re)built on top of the `development` branch.

**Tech Stack:** C++20, AMReX (AMR + GPU `ParallelFor`), Quokka `QuokkaSimulation`/`AMRSimulation` template framework, CMake + CTest, the `quokka` CLI wrapper for configure/build/run.

---

## Orientation (read first — you have no prior context)

This is the [Quokka](https://github.com/quokka-astro/quokka) radiation-hydrodynamics code.

- **Repo root** = the directory printed by `pwd` at session start (a git worktree, e.g. `/Users/u1149259/softwares/quokka/quokka-local4`). Call it `<REPO_ROOT>`. Do **not** `cd`; pass `--root <REPO_ROOT>` to every `quokka` command.
- **You are already on branch `chong/claude/stellar-evolution-framework`** (created off `development`). Confirm with `git branch --show-current`. If not on it: `git checkout chong/claude/stellar-evolution-framework`.
- **Invoke the `quokka-dev` skill** (via the Skill tool) before building — it documents the `quokka` CLI. **Invoke the `python-envs` skill** if Python is needed (we build with `-DQUOKKA_PYTHON=OFF`, so it usually isn't).
- **Design doc** (full rationale): `docs/superpowers/specs/2026-06-15-stellar-evolution-framework-design.md`. Read it if a decision is unclear.

### How "tests" work here (important — not pytest)
There is **no unit-test harness for device functions**. A "test" is a problem executable in `src/problems/<Name>/` that runs a simulation and returns process exit status `0` on success, non-zero on failure, registered with CTest via `add_test`. So our TDD loop is: write the validating problem, build it, run it, watch it fail/pass. For the framework headers, "verification" = the code compiles when a problem instantiates it.

### Build / run commands (used throughout)
```bash
# Configure once (and after any branch/submodule change). 3d preset (Star particles need AMREX_SPACEDIM==3).
quokka config -d 3d --delete --source -- --root <REPO_ROOT> -DQUOKKA_PYTHON=OFF

# Build a target (problem name == CMake target == executable)
quokka build  -d 3d <TargetName> --source -- --root <REPO_ROOT>

# Clean stale output, then run
quokka clean  --root <REPO_ROOT>
quokka run    -d 3d <TargetName> --source -- --root <REPO_ROOT>
```
Run `quokka config` **now**, before Task 1, so the build tree exists.

### Critical fact: the one-timestep lag
In `src/simulation.hpp`, inside `AMRSimulation::evolve()`, `particleRegister_.updateParticleProperties(...)` (the stellar update) runs **before** `particleMeshInteraction(...)` (accretion, which writes `mdot`). So at step *N* the stellar update sees `M`/`mdot` from the end of step *N−1*. The per-step mismatch is `~ mdot·dt/M ≲ 1e-4`, far below the test's 1–2% tolerance. Do **not** reorder the core sequence; the tolerance absorbs it.

### Refinement vs the design doc
The design doc described auto-composing the particle component count from `model::nExtraReal`. We keep that (it is cycle-free because the toy model's pure laws live in a header with no particle dependency). Two simplifications you will see below, both deliberate:
- **`Star` allows_creation = false.** On-the-fly star formation from gas is out of scope; the test pre-places one particle. This avoids porting any star-formation code. (`Rad`/`CICRad` already use `allows_creation = false`, so this path is proven.)
- **Uniform grid, `max_level = 0`** in the test — simpler particle placement and no `refineGrid` needed. The R(M)/L(M,ṁ) assertions don't depend on Bondi-rate accuracy, so coarse resolution is fine; the Bondi cross-check is informational.

---

## File Map

**New files**
- `src/particles/stellar_models.hpp` — `StellarModel_Traits` + `ToyStellarModel` (pure analytic laws; no particle dependency).
- `src/particles/starparticle_radiation.hpp` — `StellarUpdate` dispatcher (particle field I/O; calls the selected model).
- `src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp` — validation problem.
- `src/problems/ParticleStarEvolution/CMakeLists.txt` — target + `add_test`.
- `inputs/ParticleStarEvolution.in` — runtime parameters.

**Modified files**
- `src/particles/particle_types.hpp` — `Star` switch/type/layout/container/IO-names/units.
- `src/particles/particle_accretion.hpp` — thread optional `mdot_index`.
- `src/particles/PhysicsParticles.hpp` — `mdotIndex_`; `updateParticleProperties(time, dt)`; register `Star` descriptor.
- `src/particles/particle_update.hpp` — thread `dt`; add `ParticlePropertyUpdateTraits<ParticleType::Star>`.
- `src/simulation.hpp` — `StarParticles` member; `InitPhyParticles` branch; pure-virtual `createInitialStarParticles`; pass `dt` to `updateParticleProperties`.
- `src/QuokkaSimulation.hpp` — declare + default-define `createInitialStarParticles`.
- `src/problems/CMakeLists.txt` — `add_subdirectory(ParticleStarEvolution)`.
- `docs/markdown/particles.md` — document the framework + toy model + test.

> **Editing existing files:** development's exact line numbers/content may differ from any snippet here. For every modified file, **read it first**, find the named anchor (an existing symbol), and mirror the closest sibling pattern (`Sink` or `StochasticStellarPop`) when adding the `Star` variant. The snippets below are the content to add, not literal patches.

---

## Task 1: Stellar-evolution model framework headers

**Files:**
- Create: `src/particles/stellar_models.hpp`
- Create: `src/particles/starparticle_radiation.hpp`

These compile on their own but aren't instantiated until a problem uses them, so the "verification" is a syntax check now and a real compile in Task 6.

- [ ] **Step 1: Create `src/particles/stellar_models.hpp`**

```cpp
#ifndef STELLAR_MODELS_HPP_
#define STELLAR_MODELS_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include <cmath>

namespace quokka
{

// Toy stellar-evolution model: stateless analytic laws.
//   R(M)      = R_sun * (M / M_sun)^0.4
//   L_star(M) = L_sun * (M / M_sun)^3.5
//   L_acc     = G * M * mdot / R
// All functions are pure (no particle, no field indices), so this header has no particle
// dependency and can be included by particle_types.hpp without a circular include.
struct ToyStellarModel {
	// Extra per-particle components this model needs beyond the base Star layout.
	// The toy model is stateless, so it needs none.
	static constexpr int nExtraReal = 0;
	static constexpr int nExtraInt = 0;

	static constexpr amrex::Real L_solar = 3.828e33; // erg/s (CODATA 2022)
	static constexpr amrex::Real radius_exponent = 0.4;
	static constexpr amrex::Real luminosity_exponent = 3.5;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto radius(amrex::Real mass) -> amrex::Real
	{
		return C::R_solar * std::pow(mass / C::M_solar, radius_exponent);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto luminosityStar(amrex::Real mass) -> amrex::Real
	{
		return L_solar * std::pow(mass / C::M_solar, luminosity_exponent);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto luminosityAcc(amrex::Real mass, amrex::Real mdot, amrex::Real radius_val) -> amrex::Real
	{
		if (radius_val <= 0.0 || mdot <= 0.0) {
			return 0.0;
		}
		return C::Gconst * mass * mdot / radius_val;
	}

	// Pure orchestrator: given current mass and accretion rate, return radius and total
	// luminosity. dt is accepted for interface symmetry with future stateful models.
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void evolve(amrex::Real mass, amrex::Real mdot, [[maybe_unused]] amrex::Real dt,
								    amrex::Real &radius_out, amrex::Real &lum_out)
	{
		radius_out = radius(mass);
		lum_out = luminosityStar(mass) + luminosityAcc(mass, mdot, radius_out);
	}
};

// Compile-time selection of the stellar-evolution model for a problem.
// Specialize this for a problem to choose a different model.
template <typename problem_t> struct StellarModel_Traits {
	using type = ToyStellarModel;
};

} // namespace quokka

#endif // STELLAR_MODELS_HPP_
```

- [ ] **Step 2: Create `src/particles/starparticle_radiation.hpp`**

This is the dispatcher that owns particle field I/O. It references `StarParticle*Idx` constants (added in Task 2) and `LuminosityGpuConstTables` (already in `particle_radiation.hpp`).

```cpp
#ifndef STARPARTICLE_RADIATION_HPP_
#define STARPARTICLE_RADIATION_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "particles/particle_radiation.hpp" // LuminosityGpuConstTables
#include "particles/particle_types.hpp"     // StarParticle*Idx, ParticleType
#include "particles/stellar_models.hpp"     // StellarModel_Traits

#if AMREX_SPACEDIM == 3

namespace quokka
{

// Framework dispatcher for per-particle stellar-evolution updates.
// Reads the particle's mass and accretion rate, calls the model selected by
// StellarModel_Traits<problem_t>, and stores the resulting radius and luminosity.
class StellarUpdate
{
      public:
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateStellarProperties(ParticleType &p, amrex::Real /*current_time*/, amrex::Real dt,
										LuminosityGpuConstTables<Nout> const & /*gpu_tables*/) noexcept
	{
		using Model = typename StellarModel_Traits<problem_t>::type;

		const amrex::Real mass = p.rdata(StarParticleMassIdx);
		const amrex::Real mdot = p.rdata(StarParticleMdotIdx);

		amrex::Real radius_val = 0.0;
		amrex::Real lum_val = 0.0;
		Model::evolve(mass, mdot, dt, radius_val, lum_val);

		p.rdata(StarParticleRadiusIdx) = radius_val;
		p.rdata(StarParticleLumIdx) = lum_val; // first luminosity slot (nGroups==1 for the toy model)
	}
};

} // namespace quokka

#endif // AMREX_SPACEDIM == 3

#endif // STARPARTICLE_RADIATION_HPP_
```

- [ ] **Step 3: Commit**

```bash
git add src/particles/stellar_models.hpp src/particles/starparticle_radiation.hpp
git commit -m "feat(particles): add modular stellar-evolution model framework + toy model"
```

---

## Task 2: `Star` particle type, layout, container, and I/O

**Files:**
- Modify: `src/particles/particle_types.hpp`

Read the file first. It defines particle enums, per-type real-component enums (e.g. `SinkParticleRealIdx`, `StochasticStellarPopParticleRealIdx`), container typedefs, the `getParticleRealComponentNames` / `getParticleIntComponentNames` functions, and a `get_units_data()` map. Mirror the `Sink` and `StochasticStellarPop` entries for every `Star` addition.

- [ ] **Step 1: Add `Star` to both particle enums**

Anchor: `enum class ParticleSwitch`. Add a new flag after the last one (use the next free bit):
```cpp
Star = bitflag<7>() // Star particles (adjust the bit index to one past the current last flag)
```
Anchor: `enum class ParticleType`. Add `Star` as a new enumerator after the last one.

- [ ] **Step 2: Include the model header for component counts**

Near the top of `particle_types.hpp`, with the other `#include`s, add:
```cpp
#include "particles/stellar_models.hpp"
```

- [ ] **Step 3: Add the `Star` real-component layout and index constants**

Place this inside the `#if AMREX_SPACEDIM == 3` particle-definitions block, next to the `Sink` definitions. The base layout is minimal; `lum` MUST be last (it expands to `nGroups` slots).
```cpp
//-------------------- Star particles --------------------

AMREX_ENUM(StarParticleDataIdx, // NOLINT
	   mass,	// Mass of the particle
	   vx,		// Velocity x
	   vy,		// Velocity y
	   vz,		// Velocity z
	   birth_time,	// Simulation time when the particle was created
	   mdot,	// Current mass accretion rate (set by the accretion module)
	   radius,	// Stellar radius (set by the stellar-evolution model)
	   lum		// Base index for luminosity components (MUST be last; expands to lum_0, lum_1, ... for nGroups)
);

constexpr int StarParticleMassIdx = static_cast<int>(StarParticleDataIdx::mass);
constexpr int StarParticleVxIdx = static_cast<int>(StarParticleDataIdx::vx);
constexpr int StarParticleVyIdx = static_cast<int>(StarParticleDataIdx::vy);
constexpr int StarParticleVzIdx = static_cast<int>(StarParticleDataIdx::vz);
constexpr int StarParticleBirthTimeIdx = static_cast<int>(StarParticleDataIdx::birth_time);
constexpr int StarParticleMdotIdx = static_cast<int>(StarParticleDataIdx::mdot);
constexpr int StarParticleRadiusIdx = static_cast<int>(StarParticleDataIdx::radius);
constexpr int StarParticleLumIdx = static_cast<int>(StarParticleDataIdx::lum);

// Number of components = base scalars (7: mass..radius) + nGroups luminosity slots + model extras.
template <typename problem_t>
constexpr int StarParticleRealComps = 7 + Physics_Traits<problem_t>::nGroups + StellarModel_Traits<problem_t>::type::nExtraReal;
template <typename problem_t> constexpr int StarParticleIntComps = StellarModel_Traits<problem_t>::type::nExtraInt;

template <typename problem_t>
using StarParticleContainer = amrex::AmrParticleContainer<StarParticleRealComps<problem_t>, StarParticleIntComps<problem_t>>;
template <typename problem_t> using StarParticleIterator = amrex::ParIter<StarParticleRealComps<problem_t>, StarParticleIntComps<problem_t>>;
```
> Note: the `7` is the count of named scalar fields before `lum`. If you add/remove a base field, update it. A `static_assert(static_cast<int>(StarParticleDataIdx::lum) == 7);` next to the constants is a good guard — add it.

- [ ] **Step 4: Add `Star` to `getParticleRealComponentNames`**

Anchor: the `if constexpr (particleType == ParticleType::Sink)` branch in `getParticleRealComponentNames`. Add a `Star` branch mirroring the `StochasticStellarPop` branch (the `true` template arg expands the trailing `lum` field to `nGroups` slots):
```cpp
} else if constexpr (particleType == ParticleType::Star) {
	return expandEnumNames<StarParticleDataIdx, StarParticleRealComps<problem_t>, true>();
}
```

- [ ] **Step 5: Add `Star` to `getParticleIntComponentNames`**

Anchor: the `Sink` branch in `getParticleIntComponentNames` (which has no integer components). Add a `Star` branch that likewise contributes no names (the toy model has `nExtraInt == 0`):
```cpp
} else if constexpr (particleType == ParticleType::Star) { // NOLINT
	// No integer components for the toy stellar model
}
```

- [ ] **Step 6: Add `Star` to `get_units_data()`**

Anchor: the `{ParticleType::Sink, ...}` entry in the `get_units_data()` map. Add a `Star` entry. Units are `{mass_pow, length_pow, time_pow, temp_pow}`. Mirror the `Sink`/`StochasticStellarPop` formatting exactly (match the key string convention used for the luminosity field by its neighbors — likely `"luminosity"`):
```cpp
{ParticleType::Star,
 {{{"mass", {1, 0, 0, 0}},
   {"vx", {0, 1, -1, 0}},
   {"vy", {0, 1, -1, 0}},
   {"vz", {0, 1, -1, 0}},
   {"birth_time", {0, 0, 1, 0}},
   {"mdot", {1, 0, -1, 0}},
   {"radius", {0, 1, 0, 0}},
   {"luminosity", {-1, 2, -3, 0}}}}},
```

- [ ] **Step 7: Syntax-only check (full compile happens in Task 6)**

This header is shared, so it compiles when any problem builds. Defer the real check; just re-read your edits for balanced braces and that `Physics_Traits` / `StellarModel_Traits` are in scope.

- [ ] **Step 8: Commit**

```bash
git add src/particles/particle_types.hpp
git commit -m "feat(particles): add Star particle type, layout, container, and I/O metadata"
```

---

## Task 3: Thread `mdot` through the accretion module

**Files:**
- Modify: `src/particles/particle_accretion.hpp`

The accretion code already updates particle mass + momentum. We add an optional `mdot_index` so the per-step accreted mass rate is written into the particle. Read the file; find `UpdateParticleMassAndMomentumInBox`, `UpdateParticleMassAndMomentum`, and `applyAccretion`.

- [ ] **Step 1: Write `mdot` in `UpdateParticleMassAndMomentumInBox`**

Add a trailing parameter `int mdot_index = -1` to the function signature. Inside the per-particle update, immediately after the three momentum components are written (look for `p.rdata(mass_index + 3) = ...`), add:
```cpp
if (mdot_index >= 0) {
	p.rdata(mdot_index) = accreted_mass / dt;
}
```
(`accreted_mass` and `dt` are already in scope there.)

- [ ] **Step 2: Forward `mdot_index` through `UpdateParticleMassAndMomentum`**

Add `int mdot_index = -1` to its signature and pass it through to the `UpdateParticleMassAndMomentumInBox(...)` call (as the new last argument).

- [ ] **Step 3: Forward `mdot_index` through `applyAccretion`**

Add `int mdot_index = -1` to its signature and pass it to the `UpdateParticleMassAndMomentum(...)` call.

- [ ] **Step 4: Commit**

```bash
git add src/particles/particle_accretion.hpp
git commit -m "feat(particles): thread optional mdot_index through accretion to record accretion rate"
```

---

## Task 4: Update-dispatch (`dt`) + descriptor registration

**Files:**
- Modify: `src/particles/particle_update.hpp`
- Modify: `src/particles/PhysicsParticles.hpp`

Read both files. On `development`, `ParticlePropertyUpdateTraits` has a default `updateProperties(p, current_time, tables)` and a container-level `updateParticleProperties(container, current_time)`, plus a `StochasticStellarPop` specialization. We (a) thread `dt` through these signatures and (b) add a `Star` specialization. **Mirror whatever structure development actually has** (base class or not); the snippets below assume the common shape.

### particle_update.hpp

- [ ] **Step 1: Include the dispatcher**

With the other includes, add:
```cpp
#include "starparticle_radiation.hpp"
```

- [ ] **Step 2: Add `dt` to the per-particle `updateProperties` signatures**

Everywhere `updateProperties(ParticleType &..., amrex::Real current_time, LuminosityGpuConstTables<Nout> const &...)` appears (the default template and the `StochasticStellarPop` specialization), insert `amrex::Real dt` after `current_time`. In the device `ParallelFor` lambda(s) that call `updateProperties(...)`, pass `dt` through (the enclosing function must accept `dt` — see next step).

- [ ] **Step 3: Add `dt` to the container-level `updateParticleProperties`**

Change every `updateParticleProperties(ContainerType *container, amrex::Real current_time)` to `(ContainerType *container, amrex::Real current_time, amrex::Real dt)`, and pass `dt` into the per-particle dispatch (the `ParallelFor` lambda capturing `dt`). Update the `StochasticStellarPop` specialization's call to `LuminosityUpdate::updateLuminosity<problem_t>(p, current_time, gpu_tables)` — it ignores `dt`, but its enclosing signatures must carry `dt`.

- [ ] **Step 4: Add the `Star` specialization**

After the `StochasticStellarPop` specialization, add (adapt the container loop to match the sibling's exact form — e.g. iterate `lev`, `ParIterType`, `GetArrayOfStructs`, `ParallelFor(np, ...)`):
```cpp
// Specialization for Star particles: dispatches to the modular stellar-evolution framework.
template <> struct ParticlePropertyUpdateTraits<ParticleType::Star> {
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time, amrex::Real dt,
									 LuminosityGpuConstTables<Nout> const &gpu_tables) noexcept
	{
		StellarUpdate::updateStellarProperties<problem_t>(p, current_time, dt, gpu_tables);
	}

	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits<Star>::updateParticleProperties()");
		if (container == nullptr) {
			return;
		}
		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		LuminosityGpuConstTables<nGroups> const gpu_tables{}; // unused by the stellar model, passed for signature parity
		for (int lev = 0; lev <= container->finestLevel(); ++lev) {
			for (typename ContainerType::ParIterType pIter(*container, lev); pIter.isValid(); ++pIter) {
				auto &particles = pIter.GetArrayOfStructs();
				auto *pData = particles().data();
				const amrex::Long np = pIter.numParticles();
				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
					auto &p = pData[idx]; // NOLINT
					ParticlePropertyUpdateTraits<ParticleType::Star>::template updateProperties<problem_t, typename ContainerType::ParticleType,
													       nGroups>(p, current_time, dt, gpu_tables);
				});
			}
		}
	}
};
```

### PhysicsParticles.hpp

- [ ] **Step 5: Add `mdotIndex_` to the descriptor base**

Anchor: `class PhysicsParticleDescriptorBase`. Add a member:
```cpp
int mdotIndex_{-1}; // Index for accretion rate (-1 if not used)
```
Add `int mdot_idx = -1` as the last constructor parameter and initialize `mdotIndex_(mdot_idx)` in the member-init list. Add a getter:
```cpp
[[nodiscard]] AMREX_FORCE_INLINE auto getMdotIndex() const -> int { return mdotIndex_; }
```

- [ ] **Step 6: Forward `mdot_idx` in the derived `PhysicsParticleDescriptor` constructor**

Add `int mdot_idx = -1` to the derived constructor and forward it to the base constructor as the last argument.

- [ ] **Step 7: Thread `dt` through `updateParticleProperties`**

Anchor: the virtual `updateParticleProperties(amrex::Real current_time)` in the base, its override in the derived descriptor, and `PhysicsParticleRegister::updateParticleProperties(amrex::Real current_time)`. Change all three to `(amrex::Real current_time, amrex::Real dt)` and pass `dt` through. The derived override body becomes:
```cpp
ParticlePropertyUpdateTraits<particleType>::template updateParticleProperties<problem_t, ContainerType>(this->container_, current_time, dt);
```

- [ ] **Step 8: Pass `mdot_index` in the accretion call**

Anchor: the derived `applySinkAccretion` override that calls `SinkAccretionUtils::applyAccretion<...>(...)`. Add `this->getMdotIndex()` as the final argument.

- [ ] **Step 9: Register the `Star` descriptor**

Anchor: the `else if constexpr (particleType == ParticleType::Sink)` branch in `registerParticleType` (or wherever descriptors are constructed). Add a `Star` branch. Argument order (from the base ctor): `mass_idx, lum_idx, birth_time_idx, death_time_idx, allows_creation, allows_destruction, evolution_stage_idx, allows_accretion, mass_at_birth_idx, mdot_idx`:
```cpp
} else if constexpr (particleType == ParticleType::Star) {
	descriptor = std::make_unique<PhysicsParticleDescriptor<ContainerType, problem_t, ParticleType::Star>>(
	    container, StarParticleMassIdx, StarParticleLumIdx, StarParticleBirthTimeIdx, -1, /*allows_creation=*/false,
	    /*allows_destruction=*/false, /*evolution_stage_idx=*/-1, /*allows_accretion=*/true, /*mass_at_birth_idx=*/-1, StarParticleMdotIdx);
}
```

- [ ] **Step 10: Add the `Star` name**

Anchor: the `switch` that maps `ParticleType` → container name string (returns e.g. `"Sink_particles"`). Add:
```cpp
case ParticleType::Star:
	return "Star_particles";
```

- [ ] **Step 11: Commit**

```bash
git add src/particles/particle_update.hpp src/particles/PhysicsParticles.hpp
git commit -m "feat(particles): thread dt through property updates and register Star descriptor"
```

---

## Task 5: Simulation wiring

**Files:**
- Modify: `src/simulation.hpp`
- Modify: `src/QuokkaSimulation.hpp`

### simulation.hpp

- [ ] **Step 1: Declare the pure-virtual creator**

Anchor: `virtual void createInitialSinkParticles() = 0;`. Add after it:
```cpp
virtual void createInitialStarParticles() = 0;
```

- [ ] **Step 2: Add the container member**

Anchor: `std::unique_ptr<quokka::SinkParticleContainer> SinkParticles;`. Add after it:
```cpp
std::unique_ptr<quokka::StarParticleContainer<problem_t>> StarParticles;
```

- [ ] **Step 3: Pass `dt` in `evolve()`**

Anchor (in `AMRSimulation::evolve()`): `particleRegister_.updateParticleProperties(cur_time);`. Change to:
```cpp
particleRegister_.updateParticleProperties(cur_time, dt_[0]);
```

- [ ] **Step 4: Add the `InitPhyParticles` branch**

Anchor: in `InitPhyParticles`, the `if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Sink)` block. Add a `Star` block (place it before the `Sink` block, mirroring the structure). Star particles are pre-placed by the user and do not support checkpoint-restart yet:
```cpp
if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Star) {
	AMREX_ASSERT(StarParticles == nullptr);
	static_assert(Physics_Traits<problem_t>::unit_system == UnitSystem::CGS, "UnitSystem must be CGS for Star particles");

	StarParticles = std::make_unique<quokka::StarParticleContainer<problem_t>>(this);
	StarParticles->SetVerbose(0);

	particleRegister_.template registerParticleType<quokka::ParticleType::Star>(StarParticles.get());

	createInitialStarParticles();
}
```
> If `registerParticleType` / the descriptor map needs the container type, confirm the `Sink` block's exact calls and mirror them. If restart handling is required by a build assertion, mirror the non-restart path only (Star restart is out of scope).

### QuokkaSimulation.hpp

- [ ] **Step 5: Declare the override**

Anchor: `void createInitialSinkParticles() override;` (inside `class QuokkaSimulation`, within the `#if AMREX_SPACEDIM == 3` block). Add after it:
```cpp
void createInitialStarParticles() override;
```

- [ ] **Step 6: Add the default (empty) definition**

Anchor: the definition of `QuokkaSimulation<problem_t>::createInitialSinkParticles()`. Add a default definition nearby:
```cpp
template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialStarParticles()
{
	const BL_PROFILE("QuokkaSimulation::createInitialStarParticles()");
	// Default: no Star particles. A problem overrides this to place particles at t=0.
	// Only effective when ParticleSwitch::Star is enabled for the problem.
}
```

- [ ] **Step 7: Build an EXISTING particle problem to verify shared-code changes**

We changed shared headers; confirm we didn't break existing particle types. `ParticleAccretion` (3d, uses `Sink`) is a good canary.
```bash
quokka config -d 3d --delete --source -- --root <REPO_ROOT> -DQUOKKA_PYTHON=OFF
quokka build  -d 3d ParticleAccretion --source -- --root <REPO_ROOT>
```
Expected: **build succeeds, no warnings about Star/mdot/dt signatures.** If it fails, the break is in the Task 3–5 signature threading — fix before continuing.

- [ ] **Step 8: Commit**

```bash
git add src/simulation.hpp src/QuokkaSimulation.hpp
git commit -m "feat(sim): wire Star particle container, init hook, and dt into the update path"
```

---

## Task 6: `ParticleStarEvolution` validation problem

**Files:**
- Create: `src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp`
- Create: `src/problems/ParticleStarEvolution/CMakeLists.txt`
- Create: `inputs/ParticleStarEvolution.in`
- Modify: `src/problems/CMakeLists.txt`

- [ ] **Step 1: Create the problem source**

`src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp`:
```cpp
/// \file testParticleStarEvolution.cpp
/// \brief Validates the toy stellar-evolution model (R(M), L(M, mdot)) for a Star particle
///        accreting from a uniform medium via the grid Bondi accretion module.

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/stellar_models.hpp"
#include "util/BC.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using amrex::Real;

struct StarEvolutionProblem {
};

// Ambient medium (matches ParticleAccretion: cold, dense, isothermal)
constexpr double T0 = 10.0;            // K
constexpr double mu = 2.33 * C::m_p;   // mean molecular weight
constexpr double cs0 = 1.882195750e4;  // sqrt(k_B T0 / mu) cm/s for T0=10 K, mu=2.33 m_p
constexpr double B0 = 1.0e-7;          // tiny background field (MHD enabled to mirror ParticleAccretion)

double rho0 = C::m_p;                  // NOLINT background density (n_H ~ 1)
double M0_in_Msun = 1.0;               // NOLINT initial particle mass
double t_end_over_t_b = 30.0;          // NOLINT run length in Bondi times

template <> struct Particle_Traits<StarEvolutionProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Star;
};

template <> struct quokka::EOS_Traits<StarEvolutionProblem> {
	static constexpr double gamma = 1.0; // isothermal
	static constexpr double cs_isothermal = cs0;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<StarEvolutionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<StarEvolutionProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // one luminosity slot
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct SimulationData<StarEvolutionProblem> {
	std::vector<Real> time;
	std::vector<Real> mass;
	std::vector<Real> mdot;
	std::vector<Real> radius;
	std::vector<Real> lum;
};

// Place a single Star particle of mass M0 at the domain center (cell-center of the origin cell).
template <> void QuokkaSimulation<StarEvolutionProblem>::createInitialStarParticles()
{
	const int lev = 0;
	using ContainerType = quokka::StarParticleContainer<StarEvolutionProblem>;
	using PType = typename ContainerType::ParticleType;

	if (amrex::ParallelDescriptor::MyProc() == amrex::ParallelDescriptor::IOProcessorNumber()) {
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();

		PType p;
		p.id() = PType::NextID();
		p.cpu() = amrex::ParallelDescriptor::MyProc();
		p.pos(0) = 0.5 * dx[0];
		p.pos(1) = 0.5 * dx[1];
		p.pos(2) = 0.5 * dx[2];

		for (int i = 0; i < quokka::StarParticleRealComps<StarEvolutionProblem>; ++i) {
			p.rdata(i) = 0.0;
		}
		p.rdata(quokka::StarParticleMassIdx) = M0_in_Msun * C::M_solar;
		p.rdata(quokka::StarParticleBirthTimeIdx) = 0.0;

		auto &particle_tile = StarParticles->GetParticles(lev)[std::make_pair(0, 0)];
		particle_tile.push_back(p);
	}
	StarParticles->Redistribute();
}

template <> void QuokkaSimulation<StarEvolutionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double rho_bg = rho0;
	const double Eint = rho_bg / mu * C::k_B * T0; // arbitrary for isothermal EOS
	const double Emag = 0.5 * B0 * B0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::density_index) = rho_bg;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<StarEvolutionProblem>::energy_index) = Eint + Emag;
	});
}

template <> void QuokkaSimulation<StarEvolutionProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange,
			   [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<StarEvolutionProblem>::mhdFirstIndex) = B_val; });
}

// Record the particle's (t, M, mdot, R, L) after every coarse step.
template <> void QuokkaSimulation<StarEvolutionProblem>::computeAfterTimestep()
{
	const int finest_level = finestLevel();
	const auto &real_data = particleRegister_.getParticleDescriptor(quokka::ParticleType::Star)->getParticleDataAtLevel(finest_level).first;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr int off = AMREX_SPACEDIM; // 3 position components precede rdata
		if (!real_data.empty()) {
			const auto &p = real_data[0];
			userData_.time.push_back(tNew_[0]);
			userData_.mass.push_back(p[off + quokka::StarParticleMassIdx]);
			userData_.mdot.push_back(p[off + quokka::StarParticleMdotIdx]);
			userData_.radius.push_back(p[off + quokka::StarParticleRadiusIdx]);
			userData_.lum.push_back(p[off + quokka::StarParticleLumIdx]);
		}
	}
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("problem");
	pp.query("M0_in_Msun", M0_in_Msun);
	pp.query("rho0", rho0);
	pp.query("t_end_over_t_b", t_end_over_t_b);

	const double M0_g = M0_in_Msun * C::M_solar;
	const double r_B = C::Gconst * M0_g / (cs0 * cs0);
	const double t_B = r_B / cs0;

	QuokkaSimulation<StarEvolutionProblem> sim;
	sim.reconstructionOrder_ = 3;
	sim.cflNumber_ = 0.3;
	sim.tempFloor_ = 10.0;
	sim.stopTime_ = t_end_over_t_b * t_B;

	sim.setInitialConditions();
	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Star)->setForceFinestLevel(true);

	sim.evolve();

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		using Model = quokka::ToyStellarModel;
		const auto &t = sim.userData_.time;
		const auto &M = sim.userData_.mass;
		const auto &mdot = sim.userData_.mdot;
		const auto &R = sim.userData_.radius;
		const auto &L = sim.userData_.lum;
		const int n = static_cast<int>(t.size());

		amrex::Print() << "\n=== Stellar-evolution validation (" << n << " samples) ===\n";
		amrex::Print() << "r_B = " << r_B << " cm, t_B = " << t_B << " s\n";

		const double tol = 2.0e-2; // 2% absorbs the one-step lag (see plan)
		int n_checked = 0;
		for (int i = 0; i < n; ++i) {
			if (!(R[i] > 0.0) || !(M[i] > 0.0)) {
				continue; // skip pre-activation samples
			}
			const double R_pred = Model::radius(M[i]);
			const double L_pred = Model::luminosityStar(M[i]) + Model::luminosityAcc(M[i], mdot[i], R[i]);

			const double R_err = std::abs(R[i] - R_pred) / R_pred;
			const double L_err = (L_pred > 0.0) ? std::abs(L[i] - L_pred) / L_pred : std::abs(L[i]);

			if (R_err > tol) {
				status += 1;
				amrex::Print() << "  FAIL[" << i << "] radius: sim=" << R[i] << " pred=" << R_pred << " rel_err=" << R_err << "\n";
			}
			if (L_err > tol) {
				status += 1;
				amrex::Print() << "  FAIL[" << i << "] lum: sim=" << L[i] << " pred=" << L_pred << " rel_err=" << L_err << "\n";
			}
			++n_checked;
		}
		amrex::Print() << "Checked " << n_checked << " active samples; tolerance = " << tol << "\n";
		if (n_checked == 0) {
			status += 1;
			amrex::Print() << "  FAIL: no active samples (particle never activated / never accreted)\n";
		}

		// Informational: linear mass growth and Bondi rate (coarse grid -> not asserted).
		if (n >= 2) {
			const double mdot_fit = (M[n - 1] - M[0]) / (t[n - 1] - t[0]);
			const double lambda = std::exp(1.5) / 4.0;
			const double Mdot_bondi = 4.0 * M_PI * rho0 * r_B * r_B * lambda * cs0;
			amrex::Print() << "Mean dM/dt = " << mdot_fit << " g/s; analytic Bondi (hydro) ~ " << Mdot_bondi << " g/s\n";
			amrex::Print() << "Mass growth over run: " << (M[n - 1] / M[0] - 1.0) * 100.0 << " %\n";
		}

		amrex::Print() << (status == 0 ? "\n=== All stellar-evolution checks passed ===\n" : "\n=== Test FAILED (status=" + std::to_string(status) + ") ===\n");
	}

	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	return status;
}
```
> If any symbol mismatches development's API (e.g. `getParticleDataAtLevel` return shape, `Physics_Indices::mhdFirstIndex`, `quokka::grid` field names, `GetParticles(lev)[std::make_pair(0,0)]`), fix by reading the matching usage in `src/problems/ParticleAccretion/testParticleAccretion.cpp` and `src/problems/ParticleStar/`-style code, and mirror it. These are the patterns this file was modeled on.

- [ ] **Step 2: Create the CMake target**

`src/problems/ParticleStarEvolution/CMakeLists.txt` (read a sibling like `src/problems/ParticleAccretion/CMakeLists.txt` and match its exact macros; this is the expected shape):
```cmake
if (AMReX_SPACEDIM EQUAL 3)
    add_executable(ParticleStarEvolution testParticleStarEvolution.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(ParticleStarEvolution)
    endif()
    add_test(NAME ParticleStarEvolution COMMAND ParticleStarEvolution ParticleStarEvolution.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
```
> Match the include/link lines exactly to the sibling — `${QuokkaObjSources}`, `${QuokkaTestParams}`, and the working directory convention vary; copy them verbatim from `ParticleAccretion/CMakeLists.txt`.

- [ ] **Step 3: Register the subdirectory**

In `src/problems/CMakeLists.txt`, add (alphabetical/near the other `Particle*` entries):
```cmake
add_subdirectory(ParticleStarEvolution)
```

- [ ] **Step 4: Create the input file**

`inputs/ParticleStarEvolution.in`:
```
# Uniform-grid Bondi accretion onto a single Star particle (toy stellar evolution validation)

# Domain: +/- 16 r_B with r_B(1 Msun, 10 K) ~ 3.75e17 cm  ->  half-size ~ 6.0e18 cm
geometry.prob_lo = -6.0e18 -6.0e18 -6.0e18
geometry.prob_hi =  6.0e18  6.0e18  6.0e18
quokka.bc = outflow outflow outflow

amr.n_cell          = 64 64 64
amr.max_level       = 0
amr.blocking_factor = 16
amr.max_grid_size   = 32

do_reflux   = 1
do_subcycle = 0
do_tracers  = 0

particles.verbose = 1

problem.M0_in_Msun     = 1.0
problem.rho0           = 1.6726e-24   # ~ m_p (n_H ~ 1)
problem.t_end_over_t_b = 30.0

plotfile_interval     = -1
checkpoint_interval   = -1
tiny_profiler.enabled = 0
```
> `quokka.bc`: use the exact boundary keyword convention development expects (read `inputs/ParticleAccretion.in`). `outflow` lets gas flow in/out for steady accretion; if development uses different tokens, match them.

- [ ] **Step 5: Build the target**

```bash
quokka config -d 3d --delete --source -- --root <REPO_ROOT> -DQUOKKA_PYTHON=OFF
quokka build  -d 3d ParticleStarEvolution --source -- --root <REPO_ROOT>
```
Expected: **build succeeds.** This is the first full instantiation of all `Star` code (container, descriptor, dispatcher, toy model). Fix compile errors here by cross-referencing the sibling problem and the headers from Tasks 1–5.

- [ ] **Step 6: Commit**

```bash
git add src/problems/ParticleStarEvolution/ inputs/ParticleStarEvolution.in src/problems/CMakeLists.txt
git commit -m "test(particles): add ParticleStarEvolution toy stellar-evolution validation problem"
```

---

## Task 7: Run and validate

**Files:** none (run + iterate)

- [ ] **Step 1: Run the test**

```bash
quokka clean --root <REPO_ROOT>
quokka run   -d 3d ParticleStarEvolution --source -- --root <REPO_ROOT>
```
Expected output ends with `=== All stellar-evolution checks passed ===` and the process exits `0`. The log should show ~tens of active samples checked, radius/lum relative errors well under 2%, a positive mean `dM/dt`, and a few-percent mass growth.

- [ ] **Step 2: Diagnose failures (if any)**

- "no active samples": the particle never accreted (`mdot` stayed 0) or `radius` stayed 0. Check that the `Star` descriptor has `allows_accretion = true` and `mdot_idx = StarParticleMdotIdx` (Task 4 Step 9), and that `mdot` is being written (Task 3). Confirm the stellar update runs (Task 4/5 wiring). Increase `t_end_over_t_b` if growth is negligible.
- radius/lum errors just above tol: confirm `nGroups == 1` and that `lum` is read from `StarParticleLumIdx` (the first lum slot). The lag should keep errors ≲ 1e-3; large errors mean a units or index bug.
- If `mdot` is read as 0 in the *first* active samples only, that's the lag — those samples have `R>0`, `mdot==0`, so `L_acc==0`; the L check still holds (`luminosityAcc` returns 0). Fine.

- [ ] **Step 3: Confirm via CTest**

```bash
quokka run -d 3d --filter '^ParticleStarEvolution$' --root <REPO_ROOT>
```
Expected: the test reports **Passed**.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix(test): make ParticleStarEvolution assertions pass"
```
(Skip if no changes were needed.)

---

## Task 8: Documentation

**Files:**
- Modify: `docs/markdown/particles.md`

- [ ] **Step 1: Add a Star / stellar-evolution section**

Append a section (the file documents particle types and feedback). Add:
```markdown
## Star Particle Type (modular stellar evolution)

Star particles (`ParticleSwitch::Star`) represent individual stars whose radius and
luminosity evolve through a **pluggable stellar-evolution model** selected at compile time.

### Particle attributes

Real components: `mass`, `vx`, `vy`, `vz`, `birth_time`, `mdot` (accretion rate, set by the
accretion module), `radius` (set by the stellar model), and `lum` (luminosity per radiation
group; occupies the last `nGroups` slots). A model may declare extra real/integer components
via `nExtraReal` / `nExtraInt`.

### Choosing a model

The model is chosen with the `StellarModel_Traits<problem_t>` trait (default
`ToyStellarModel`). A model is a struct of GPU device functions; the dispatcher
`StellarUpdate::updateStellarProperties` reads the particle's `mass` and `mdot`, calls the
model, and stores `radius` and `lum` once per coarse step (operator-split, after accretion).

### Toy model

`ToyStellarModel` is stateless:

- Radius: $R = R_\odot (M/M_\odot)^{0.4}$
- Stellar luminosity: $L_\star = L_\odot (M/M_\odot)^{3.5}$
- Accretion luminosity: $L_\mathrm{acc} = G M \dot{M} / R$
- Stored luminosity: $L = L_\star + L_\mathrm{acc}$

### Validation test

`ParticleStarEvolution` places one Star particle in a uniform medium and lets it accrete via
the grid Bondi accretion module (small-mass regime, $\dot{M}\approx$ const). It asserts, each
step, that the particle's stored radius and luminosity match the closed-form laws above
within ~2% (the tolerance absorbing the one-timestep lag between accretion and the stellar
update), and reports the mean accretion rate against the analytic Bondi value.
```

- [ ] **Step 2: Commit**

```bash
git add docs/markdown/particles.md
git commit -m "docs: document modular stellar-evolution framework, toy model, and validation test"
```

---

## Task 9: Final review

- [ ] **Step 1: Re-run the canary + new test together**

```bash
quokka build -d 3d ParticleAccretion       --source -- --root <REPO_ROOT>
quokka build -d 3d ParticleStarEvolution   --source -- --root <REPO_ROOT>
quokka clean --root <REPO_ROOT>
quokka run   -d 3d --filter '^(ParticleAccretion|ParticleStarEvolution)$' --root <REPO_ROOT>
```
Expected: both pass. (ParticleAccretion proves the shared-code threading didn't regress existing particles.)

- [ ] **Step 2: Lint the changed files**

```bash
scripts/tidy.sh build changed
```
Address real warnings in files you created/modified (ignore the AMReX-include false positives the editor's language server shows).

- [ ] **Step 3: Summarize**

Report in chat (1–2 sentences) what was built and that both tests pass, then offer to open a PR per the `quokka-dev` skill's "Opening a PR" workflow (do not push without explicit user approval).

---

## Self-Review (already performed by plan author)

- **Spec coverage:** framework (Task 1), toy model (Task 1), minimal layout + extras seam (Task 2), mdot wiring (Task 3), dt threading + dispatch (Task 4), sim wiring (Task 5), test with the four design assertions — R(M), L(M,ṁ) asserted; M(t) linear growth + Bondi rate reported informationally given the deliberately coarse uniform grid (Task 6–7), docs (Task 8). All design sections map to a task.
- **Deliberate deviations from the design doc** (documented in "Orientation"): `allows_creation=false` (no formation), uniform `max_level=0` grid (assertions 3–4 informational). These reduce scope without weakening the R(M)/L(M,ṁ) validation, which is the framework's actual contract.
- **Type consistency:** `StarParticleMassIdx/MdotIdx/RadiusIdx/LumIdx/BirthTimeIdx`, `StarParticleRealComps<>`, `StellarModel_Traits<>::type`, `ToyStellarModel::{radius,luminosityStar,luminosityAcc,evolve}`, and `StellarUpdate::updateStellarProperties` are used identically across Tasks 1, 2, 4, and 6.
```
