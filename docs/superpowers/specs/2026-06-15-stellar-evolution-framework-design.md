# Modular Stellar-Evolution Framework + Validation Test — Design

**Date:** 2026-06-15
**Branch:** `chong/claude/stellar-evolution-framework` (off `development`)
**Status:** Approved design; implementation plan to follow.

## 1. Motivation

The `chong/particles/starparticles-copy` branch introduced a `Star` particle type whose
stellar physics lives in `src/particles/starparticle_radiation.hpp` as a single monolithic
`StellarUpdate::updateStellarProperties` routine: a deuterium-burning state machine,
Lane–Emden central-condition tables, TOMS-748 EOS/β solvers, and Tout (1996) ZAMS fits.
There is no seam to swap the physics, and the existing `ParticleStar` test only checks mass
conservation and field sanity — it never validates against an analytic solution.

This work **rebuilds the Star particle from scratch off `development`** as a *modular*
stellar-evolution framework: the evolution model is selected at compile time (like
`EOS_Traits` / `Physics_Traits`), with a simple analytic **toy model** as the default and
the seam to add realistic models later. A new validation test, modeled on
`ParticleAccretion`, checks the toy model's radius and luminosity against closed-form laws.

## 2. Goals / Non-goals

**Goals**
- A compile-time-selectable stellar-evolution model framework, GPU-friendly.
- A stateless `ToyStellarModel`: `R(M)`, `L_star(M)`, plus accretion luminosity.
- A minimal, clean Star particle layout with an extension seam for stateful models.
- A new test (`ParticleStarEvolution`) validating the toy model against analytic solutions.
- Updated `docs/markdown/particles.md`.

**Non-goals**
- Re-implementing the detailed protostellar model (future work; the seam will exist).
- Radiation deposition of the stored luminosity (validate the stored `lum` field only).
- AMR subcycling correctness for stellar updates (inherits current behavior).

## 3. Framework architecture

Mirrors existing Quokka trait idioms. A **stellar-evolution model** is a struct of static
`AMREX_GPU_DEVICE` functions in two layers:

- **Pure laws** (also callable directly from tests/unit checks):
  `radius(M)`, `luminosityStar(M)`, `luminosityAcc(M, mdot, R)`.
- **Orchestrator**: `evolve(p, dt)` — reads the particle's fields, advances any internal
  state, and writes `radius` and `lum` back to the particle.

Model selection trait (default provided):

```cpp
template <typename problem_t> struct StellarModel_Traits {
    using type = quokka::ToyStellarModel; // default
};
```

The dispatcher `StellarUpdate::updateStellarProperties`, reached through
`ParticlePropertyUpdateTraits<ParticleType::Star>`, calls
`StellarModel_Traits<problem_t>::type::evolve(p, dt)`.

### Modularity seam for stateful models

Each model declares:

```cpp
static constexpr int nExtraReal = 0;  // extra real components
static constexpr int nExtraInt  = 0;  // extra integer components
```

plus named offsets into those extra slots. The Star particle component counts are computed
as `base + nGroups + model::nExtraReal` (real) and `baseInt + model::nExtraInt` (int), all
`constexpr`. `ToyStellarModel` declares `0` extras. A future `ProtostellarModel` would add
`mdeut`, `n` (real) and `burnState` (int) without touching the framework or the base layout.

## 4. `ToyStellarModel` (default, stateless)

| Quantity | Formula |
|----------|---------|
| Radius | `R = R_sun * (M/M_sun)^0.4` |
| Stellar luminosity | `L_star = L_sun * (M/M_sun)^3.5` |
| Accretion luminosity | `L_acc = G * M * mdot / R` |
| Total stored `lum` | `L_star + L_acc` |

`evolve(p, dt)`: read `M = p.mass`, `mdot = p.mdot`; `R = radius(M)`;
`lum = luminosityStar(M) + luminosityAcc(M, mdot, R)`; store `p.radius = R`, `p.lum = lum`.
No internal state; `nExtraReal = nExtraInt = 0`. `birth_time` is set at particle creation.

## 5. Star particle plumbing (rebuilt from `development`)

| File | Change |
|------|--------|
| `src/particles/particle_types.hpp` | `ParticleSwitch::Star`, `ParticleType::Star`; real layout `{mass, vx, vy, vz, birth_time, mdot, radius, lum[nGroups]}` (+ model extras); `StarParticleContainer<problem_t>` / iterator; comp counts; I/O names; units map entry. |
| `src/particles/particle_accretion.hpp` | Thread an optional `mdot_index` through `UpdateParticleMassAndMomentumInBox` / `UpdateParticleMassAndMomentum` / `applyAccretion` (writes `accreted_mass/dt` into the particle's `mdot`). |
| `src/particles/PhysicsParticles.hpp` | `mdotIndex_` on the descriptor base; `updateParticleProperties(time, dt)` signature; register the `Star` descriptor (`mass`, `lum`, `birth_time`, `death_time`, `mdot` indices; `allows_creation = true`, `allows_accretion = true`); `Star` name + units. |
| `src/particles/particle_update.hpp` | Thread `dt`; `ParticlePropertyUpdateTraits<ParticleType::Star>` → `StellarUpdate::updateStellarProperties`. |
| `src/particles/starparticle_radiation.hpp` (new, clean) | Framework: `StellarModel_Traits`, `ToyStellarModel`, `StellarUpdate` dispatcher. |
| `src/simulation.hpp` | `StarParticles` container member; `InitPhyParticles` branch (CGS assert, register, `createInitialStarParticles`); pure-virtual `createInitialStarParticles`; `updateParticleProperties(cur_time, dt_[0])`. |
| `src/QuokkaSimulation.hpp` | `createInitialStarParticles` override declaration + default (empty) definition. |

### Particle creation in the test

`InitFromAsciiFile` cannot carry integer components, so `createInitialStarParticles` inserts
the single particle programmatically (construct the AoS particle, set `pos`, `rdata`, `idata`,
push onto the level-0 / finest tile on the IO processor, then `Redistribute`). The particle
is forced to the finest level via `setForceFinestLevel(true)`.

## 6. Validation test `ParticleStarEvolution` (modeled on `ParticleAccretion`)

**Setup**
- Uniform medium via `uniform_density` (constant background → clean Bondi rate).
- One Star particle, `M0 = 1 M_sun`, at the domain center; center refined; forced to finest level.
- `Physics_Traits`: hydro on, self-gravity on, MHD on with tiny `B0 = 1e-7` (mirrors
  `ParticleAccretion`; the Bondi cross-check uses the MHD-aware fast-magnetosonic form),
  radiation off, `nGroups = 1`. Uses the default `ToyStellarModel` (no trait specialization).
- `computeAfterTimestep` records `(t, M, mdot, R, L)` from the particle each coarse step.

**Reference parameters** (T = 10 K, μ = 2.33 mₚ, n_H ≈ 1, M0 = 1 M⊙):
`cs ≈ 1.88e4 cm/s`, `r_B ≈ 0.12 pc`, Bondi `mdot ≈ 6.2e16 g/s ≈ 9.8e-10 M⊙/yr`,
`L_acc ≈ 1.2e32 erg/s ≈ 0.03 L⊙`, `L_star = L⊙`. The 1%-mass-growth time ≈ 10 Myr ≈ ~100
coarse steps, so a ~100-step run stays in the small-mass / constant-`mdot` regime.

**Assertions** (relative tolerance ~1–2%, which absorbs the one-step lag — see §7):
1. `R_sim(t) ≈ R_sun * (M_sim/M_sun)^0.4`.
2. `L_sim(t) ≈ L_sun * (M_sim/M_sun)^3.5 + G * M_sim * mdot_sim / R_sim`.
3. `M_sim(t) ≈ M0 + mdot * t` (linear growth, Bondi small-mass).
4. `mdot_sim ≈` analytic Bondi rate (cross-check against the `ParticleAccretion` formula).

**Artifacts**
- `src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp`
- `src/problems/ParticleStarEvolution/CMakeLists.txt` (+ `add_test`)
- `inputs/ParticleStarEvolution.in` (~64 r_B box, refine center, ~100 coarse steps)
- Register the subdirectory in `src/problems/CMakeLists.txt`.

## 7. The one-step lag

In `evolve()`, `updateParticleProperties` (stellar update) runs **before**
`particleMeshInteraction` (accretion, which writes `mdot`). So the `radius`/`lum` stored at
the end of step *N* were computed from the `M`, `mdot` at the **start** of step *N* (i.e.
after step *N−1*'s accretion), while `computeAfterTimestep` reads the post-step-*N* `M`,
`mdot`. The mismatch per step is `~ mdot·dt / M ≲ 1e-4`, far inside the 1–2% tolerance.
We deliberately do **not** reorder the core accretion/update sequence (out of scope, risky);
the loose tolerance handles it. (A rigorous alternative — comparing `R(t_N)` against
`radius(M(t_{N-1}))` from recorded history — is noted but not the default.)

## 8. Validation procedure

```
quokka config  -d 3d --delete --source -- --root <REPO_ROOT> -DQUOKKA_PYTHON=OFF
quokka build   -d 3d ParticleStarEvolution --source -- --root <REPO_ROOT>
quokka clean   --root <REPO_ROOT>
quokka run     -d 3d ParticleStarEvolution --source -- --root <REPO_ROOT>
```
Expect process exit status 0 (all assertions pass).

## 9. Risks / open points

- **Particle layout churn**: changing the Star real/int layout touches I/O names + units;
  keep base layout minimal and stable, push variability into model extras.
- **Self-gravity + accretion** interplay in a uniform medium: mirror `ParticleAccretion`'s
  proven configuration to avoid surprises.
- **MHD on**: keeps the test close to `ParticleAccretion`; if it complicates the Bondi
  cross-check, falling back to pure hydro is acceptable (assertions 1–2 are independent of
  the accretion rate's exact value).
