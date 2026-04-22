# Plan: Per-Level Particle Drift and Kick During AMR Subcycling (Option B)

## Overview

Currently, particle leapfrog integration (kick-drift-kick) happens at the **global (coarsest-level) timestep** `dt_[0]`, regardless of which AMR level a particle resides on. Fine-level particles that should take multiple smaller substeps instead get one big step with the coarse `dt_[0]`.

This document describes **Option B**: keep the single Poisson solve per global step, but move particle drift and kick into `timeStepWithSubcycling` so each level integrates its particles with its own `dt_[lev]`. The acceleration field is computed once from the global Poisson solve and reused for all substep kicks.

## Lessons from Nyx

Nyx (`/Users/benwibking/amrex_codes/Nyx`) implements per-level particle integration with AMR subcycling. Key patterns:

### Ghost and Virtual Particles

Nyx uses two particle categories for cross-level communication during subcycling:

- **Virtual particles** (`setup_virtual_particles`): Fine-level particles projected down to coarser levels. This lets the coarse level include the gravitational influence of fine-level particles during its (longer) timestep. Created once at the start of each coarse timestep; removed in `post_timestep` via `remove_virtual_particles`.

- **Ghost particles** (`setup_ghost_particles`, with `ghost_width`): Coarse-level particles projected up to finer levels. Fine-level substeps can then see and kick/drift these ghost particles, ensuring particles near coarse-fine boundaries are consistently evolved. Created with width `ghost_width = ncycle + stencil_deposition_width`. Removed in `post_timestep` only on the final iteration (`iteration == ncycle`) via `remove_ghost_particles`.

**Quokka does not currently use virtual or ghost particles.** Its particle redistribution (`Redistribute`) moves particles to the correct level after each step, but does not create representative copies at other levels. For Option B without virtual/ghost particles, particles near coarse-fine boundaries will be integrated with the timestep of whichever level they currently reside on. This is acceptable for the initial implementation but will cause O(dt) time-centering errors at coarse-fine boundaries. Adding ghost/virtual particles is a future enhancement.

### The `where_width` and `grav_n_grow` Parameters

Nyx computes these per-subcycle-iteration quantities to control ghost cell widths:

```
ghost_width  = ncycle + stencil_deposition_width    // constant across iterations
where_width  = ghost_width + (1 - iteration) - 1     // shrinks each iteration
grav_n_grow  = ghost_width + (1 - iteration) + (iteration - 1) + stencil_interpolation_width
             = ghost_width + stencil_interpolation_width   // actually constant!
```

- `ghost_width`: How many cells of ghost particles to create at the coarse level for use by fine levels. Depends on the subcycling ratio `ncycle`.
- `where_width`: After drifting, particles are checked against a grid assignment with this width. On iteration 1, particles have just been drifted so they could be up to `ghost_width` cells out; on later iterations they've already been assigned so they're at most 1 cell out. The `(1 - iteration)` term decreases the check width as iterations progress.
- `grav_n_grow`: Ghost cells needed on the acceleration MultiFab so that CIC interpolation at particle positions is valid. This is constant because the `(1-iteration)` and `(iteration-1)` terms cancel.

### Post-Drift Grid Assignment in `moveKickDrift`

After Nyx drifts particles at fine levels (`lev > 0 && sub_cycle`), it re-assigns particles to their grid cells using `assign_grid(p, lev, lev, where_width)`. Any ghost particle that has moved out of the valid ghost region is marked for removal (`id = -1`). This is critical because:

1. After drifting, a particle may have crossed from one grid/tile to another or from one level to another.
2. The `where_width` parameter tells the grid assignment how many cells of "slop" are acceptable — if a particle is farther out than `where_width` cells, it's outside the valid region and must be removed.
3. This check is only done at `lev > 0` when subcycling is active.

### Redistribution Timing in `post_timestep`

Nyx's `post_timestep(iteration)` (in `Nyx.cpp:1447`):

```
1. Remove virtual particles at this level
2. Remove ghost particles if iteration == ncycle (final substep only)
3. Redistribute active particles:
   - if (iteration < ncycle && level < finest_level) or (level == 0):
       Redistribute(level, finestLevel, iteration)   // nGrow = iteration!
   - if finest_level == 0:
       RedistributeLocal(level, finestLevel, iteration)
```

Key detail: **`Redistribute` is called with `nGrow = iteration`** — the number of ghost cells for redistribution equals the current subcycle iteration number. This allows particles that have been drifted by up to `iteration` substeps to be correctly assigned to their owning grid, accounting for the fact that particles move progressively farther from their starting position as subcycling proceeds.

### The `moveKick` (Second Half-Kick) and Ghost Particle Handling

In `advance_hydro_plus_particles` (`Nyx_advance.cpp:394-412`):

```
// After new-time Poisson solve:
for (int lev = level; lev <= finest_level_to_advance; lev++) {
    // Kick active particles with NEW acceleration
    theActiveParticles()[i]->moveKick(grav_vec_new, lev, dt, a_new, a_half);

    // Virtual particles will be recreated, so we need not kick them.

    // Ghost particles need to be kicked except during the final iteration.
    if (iteration != ncycle)
        theGhostParticles()[i]->moveKick(grav_vec_new, lev, dt, a_new, a_half);
}
```

**Ghost particles are NOT kicked on the final iteration** because on the final substep, the real coarse-level particle will get the correct second-half kick with the new acceleration. Kicking the ghost on the final iteration would double-count.

### Quokka's Current Redistribution (for comparison)

In `timeStepWithSubcycling` (`simulation.hpp:2253-2264`):

```cpp
if ((iteration < nsubsteps[lev]) || (lev == 0)) {
    if (lev == 0) {
        redistribute_ngrow = 0;
    } else {
        redistribute_ngrow = iteration;
    }
    particleRegister_.redistribute(lev, redistribute_ngrow);
}
```

This already follows the Nyx pattern: `nGrow = iteration` for `lev > 0`, `nGrow = 0` for `lev == 0`. No change is needed here.

---

## Current Architecture

### Main timestep loop (`evolve()`, `src/simulation.hpp:~1437-1487)

```
1. kickParticlesAllLevels(dt_[0])            // first half-kick (line 1444)
2. timeStepWithSubcycling(0, cur_time, 1)     // hydro advance, all levels (line 1454)
3. driftParticlesAllLevels(dt_[0], finest)    // drift (line 1461)
4. ellipticSolveAllLevels(dt_[0])             // Poisson solve + grav accel on gas (line 1467)
5. kickParticlesAllLevels(dt_[0])             // second half-kick (line 1473)
6. updateParticleProperties / particleMeshInteraction / destroyParticles (lines 1478-1485)
```

**Important API note:** `kickParticles` (`PhysicsParticles.hpp:491`) internally multiplies by 0.5:
```cpp
p.rdata(comp) += 0.5 * dt * static_cast<amrex::ParticleReal>(acc_comp);
```
So `kickParticlesAtLevel(lev, dt_[0], accel)` performs a **half-kick** (`vel += 0.5 * dt * accel`). The current code passes the full `dt_[0]` to get a half-kick, and calls it twice (once before and once after the drift) to complete the full KDK cycle.

### Current time-centering of the Poisson solve

In the current code, the Poisson solve at step n computes `phi` from the **post-hydro, post-drift** state (density and particle positions after the hydro advance and particle drift at step n). The first half-kick at step n uses `phi` from step n-1's Poisson solve (the "old" potential), and the second half-kick uses `phi` from step n's Poisson solve (the "new" potential). This gives time-centered kicks: the first half uses `a^(n-1/2)` and the second half uses `a^(n+1/2)`, averaging to `a^n` at the drift midpoint.

### Key constraint

`calculateGpotAllLevels()` (`simulation.hpp:1706`) explicitly aborts if `do_subcycle == 1`:

```cpp
if (do_subcycle == 1) { // not supported
    amrex::Abort("Poisson solve is not supported when AMR subcycling is enabled! ...");
}
```

This means the Poisson solve is always done once per global step, and `dt_[lev] == dt_[0]` for all levels in practice today. Option B does not change this — we keep one Poisson solve, but restructure particle integration to be level-aware so the code is correct when subcycling is eventually enabled.

### Particle API (current)

| Method | Location | Current scope |
|--------|----------|---------------|
| `driftParticles(lev_min, lev_max, dt)` | `PhysicsParticles.hpp:434` | Loops `lev_min..lev_max`, single `dt` |
| `kickParticles(lev, dt, accel)` | `PhysicsParticles.hpp:462` | Single level, single `dt`; internally applies `0.5 * dt` |
| `driftParticlesAllLevels(dt, lev_max)` | `PhysicsParticles.hpp:1106` | Calls `driftParticles(0, lev_max, dt)` |
| `kickParticlesAtLevel(lev, dt, accel)` | `PhysicsParticles.hpp:1119` | Calls `kickParticles(lev, dt, accel)` per type |
| `kickParticlesAllLevels(dt)` | `simulation.hpp:1972` | Loops all levels, computes accel, kicks |

### Subcycling structure (`timeStepWithSubcycling`, `simulation.hpp:2136)

```
timeStepWithSubcycling(lev, time, iteration):
    regrid if needed
    tOld_[lev] = tNew_[lev]
    tNew_[lev] += dt_[lev]          // critical: before advanceAtLevel
    advanceSingleTimestepAtLevel(lev, ...)
    for i = 1..nsubsteps[lev+1]:
        timeStepWithSubcycling(lev+1, ...)     // recurse into finer level
    reflux + AverageDown + FixupState
    redistribute particles at level lev
```

No particle drift or kick is currently performed inside this function.

---

## Proposed Architecture

### Poisson solve timing and time-centering

The most significant design decision is **when to compute the gravitational potential** relative to the per-level particle kicks. There are two approaches:

**Approach A (recommended): Compute phi before subcycling, use for all kicks.**

Move `calculateGpotAllLevels()` before `timeStepWithSubcycling()`. The potential is computed from the current state (pre-hydro, pre-drift), which is the same as the state at the end of the previous step. Both half-kicks at all levels use this single phi. Gas gravity (`gravAccelAllLevels`) is applied after subcycling using the same phi.

Trade-offs:
- Particle kicks use phi computed from pre-hydro density rather than the time-centered phi used in the current code. This introduces an O(dt^2) local error per step, the same order as the leapfrog truncation error itself.
- Gas gravity uses phi from pre-hydro density rather than post-hydro density (a change from the current code). This introduces an O(dt) local error for gas gravity, comparable to the existing operator-splitting error.
- The Poisson solve uses pre-drift particle positions for mass deposition (a change from the current code, which uses post-drift positions). The O(v*dt) position lag is comparable to the lag in the current code's first half-kick, which uses phi from the previous step's post-drift positions.
- This approach allows the full KDK structure inside subcycling, which is essential for correct fine-level particle integration.

**Approach B (alternative, future enhancement): Use phi_prev for first half-kick, compute phi_new after subcycling for second half-kick.**

Keep `calculateGpotAllLevels()` after subcycling (current position). The first half-kick uses `phi` from the previous step's Poisson solve (already stored in `phi[lev]`). After subcycling and the Poisson solve, the second half-kick uses phi_new. This preserves the time-centering of the current code.

Challenges:
- The second half-kick cannot happen inside the subcycling loop (phi_new isn't available yet), breaking the per-level KDK structure for fine levels. Fine-level particles would drift with a half-kicked velocity between substeps, requiring a "combined kick" trick (second half-kick of substep i + first half-kick of substep i+1 = full kick with phi_prev) followed by a correction kick after the Poisson solve.
- Requires storing the old acceleration field or recomputing it for the correction.
- More complex implementation with no clear accuracy benefit for Option B (since both approaches have O(dt^2) error from the single Poisson solve).

**Approach A is recommended** for its simplicity and because it allows the natural KDK structure inside subcycling. Approach B can be revisited if higher accuracy is needed.

### New flow (Approach A)

```
evolve():
    // (removed: global kickParticlesAllLevels)

    // Compute gravitational potential once per global step.
    // phi is computed from the current (pre-hydro, pre-drift) state.
    if (self_gravity && poisson_supercycle_ready):
        calculateGpotAllLevels()

    timeStepWithSubcycling(0, cur_time, 1):
        kickParticlesAtLevel(lev, dt_[lev], accel_lev)   // first half-kick
        tOld_[lev] = tNew_[lev];                          // update time levels
        tNew_[lev] += dt_[lev];                           // update time levels
        advanceSingleTimestepAtLevel(lev)                  // hydro advance
        driftParticles(lev, lev, dt_[lev])                // per-level drift
        // recurse into finer levels (nsubsteps[lev+1] sub-steps)
        for i = 1..nsubsteps[lev+1]:
            timeStepWithSubcycling(lev+1, time + (i-1)*dt_[lev+1], i):
                kickParticlesAtLevel(lev+1, dt_[lev+1], accel_lev+1)  // first half-kick
                tOld_[lev+1] = tNew_[lev+1]
                tNew_[lev+1] += dt_[lev+1]
                advanceSingleTimestepAtLevel(lev+1)
                driftParticles(lev+1, lev+1, dt_[lev+1])
                kickParticlesAtLevel(lev+1, dt_[lev+1], accel_lev+1)  // second half-kick
                // ... further recursion ...
        kickParticlesAtLevel(lev, dt_[lev], accel_lev)   // second half-kick
        reflux / AverageDown / FixupState
        redistribute particles at level lev (nGrow = iteration for lev > 0)

    // (removed: global driftParticlesAllLevels)
    // (removed: second global kickParticlesAllLevels)

    // Apply gravity to gas using phi computed before subcycling.
    // This uses the same phi as the particle kicks.
    gravAccelAllLevels(dt_[0])
    updateParticleProperties / particleMeshInteraction / destroyParticles
```

### Leapfrog KDK scheme with subcycling

The leapfrog (kick-drift-kick) scheme must be adapted so that each level does its own KDK with its own `dt_[lev]`. For a level that subcycles with ratio `nsubsteps`, the particle sees:

```
Level 0 (coarse):  K(dt_0) -- drift(dt_0) -- K(dt_0)
Level 1 (fine):    n substeps, each: K(dt_1) -- drift(dt_1) -- K(dt_1)
```

where `K(dt)` denotes a half-kick: `vel += 0.5 * dt * accel` (the `kickParticles` function internally applies the 0.5 factor). Both half-kicks use the acceleration field from the single Poisson solve computed before subcycling begins.

---

## Detailed Changes

### Change 1: Extract acceleration-field helper from `kickParticlesAllLevels`

**File:** `src/simulation.hpp`

**What:** Factor out the acceleration-field computation (lines 1983–2062 of `kickParticlesAllLevels`) into a new method `computeAccelerationAtLevel(int lev)` that returns an `amrex::MultiFab` of the cell-centered acceleration field at level `lev`.

**Current code (lines 1983–2063):**
```cpp
template <typename problem_t> void AMRSimulation<problem_t>::kickParticlesAllLevels(const amrex::Real dt)
{
    // ... skip check ...
    for (int lev = 0; lev <= finest_level; ++lev) {
        // 1. Build phi_extended with ghost cells (lines 1994-2037)
        // 2. Compute accel_cc from phi_extended gradient (lines 2043-2057)
        // 3. Kick particles: particleRegister_.kickParticlesAtLevel(lev, dt, accel_cc);
    }
}
```

**New helper:**
```cpp
#if AMREX_SPACEDIM == 3
template <typename problem_t>
amrex::MultiFab AMRSimulation<problem_t>::computeAccelerationAtLevel(int lev);
#endif
```

This method should:
1. Create `phi_extended` with `nghost_phi` ghost cells on `boxArray(lev)` (see Change 8 for dynamic `nghost_phi`).
2. Fill it from `phi[lev]` using `FillBoundary` and `PhysBCFunct` (exactly as in lines 2005–2037).
3. Compute `accel_cc` from the gradient of `phi_extended` (exactly as in lines 2043–2057).
4. Return `accel_cc`.

The existing `kickParticlesAllLevels` can then be simplified to:
```cpp
template <typename problem_t> void AMRSimulation<problem_t>::kickParticlesAllLevels(const amrex::Real dt)
{
    if (!particleRegister_.HasMassiveParticles()) { return; }
    for (int lev = 0; lev <= finest_level; ++lev) {
        auto accel_cc = computeAccelerationAtLevel(lev);
        particleRegister_.kickParticlesAtLevel(lev, dt, accel_cc);
    }
}
```

**Why:** This allows `timeStepWithSubcycling` to call `computeAccelerationAtLevel` independently for any level, without the full loop.

### Change 2: Move drift and kick into `timeStepWithSubcycling`

**File:** `src/simulation.hpp`, function `timeStepWithSubcycling` (starts at line 2136)

**What:** Insert particle drift and kick around the hydro advance and subcycling recursion.

**Important:** `kickParticles` internally applies a factor of 0.5 (half-kick). To perform a half-kick with timestep `dt_[lev]`, call `kickParticlesAtLevel(lev, dt_[lev], accel_cc)`. Do **not** pass `0.5 * dt_[lev]` or `dt_[lev] / 2`, which would produce a quarter-kick.

**Insert before `tOld_[lev] = tNew_[lev]` (line 2194) and `advanceSingleTimestepAtLevel`:**

```cpp
#if AMREX_SPACEDIM == 3
if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
    if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
        // First half-kick: vel += 0.5 * dt_[lev] * accel
        // (kickParticlesAtLevel internally multiplies by 0.5, so pass dt_[lev])
        if (particleRegister_.HasMassiveParticles()) {
            auto accel_cc = computeAccelerationAtLevel(lev);
            particleRegister_.kickParticlesAtLevel(lev, dt_[lev], accel_cc);
        }
    }
}
#endif
```

**Insert after `advanceSingleTimestepAtLevel(lev, ...)` (line 2198) and before the recursion (line 2210):**

```cpp
#if AMREX_SPACEDIM == 3
if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
    // Drift particles at this level with level-appropriate timestep
    if (particleRegister_.HasMassiveParticles()) {
        particleRegister_.driftParticles(lev, lev, dt_[lev]);
    }
}
#endif
```

**Insert after the recursion loop (after line 2238, before reflux):**

```cpp
#if AMREX_SPACEDIM == 3
if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
    if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
        // Second half-kick: vel += 0.5 * dt_[lev] * accel
        if (particleRegister_.HasMassiveParticles()) {
            auto accel_cc = computeAccelerationAtLevel(lev);
            particleRegister_.kickParticlesAtLevel(lev, dt_[lev], accel_cc);
        }
    }
}
#endif
```

**Placement of `tOld_`/`tNew_` updates:** The `tOld_[lev]` and `tNew_[lev]` updates (lines 2194-2195) must remain immediately before `advanceSingleTimestepAtLevel`, as the hydro advance depends on these time levels for reconstruction. The particle kick and drift do not depend on `tOld_`/`tNew_`, so the first half-kick can be placed before the time level updates without affecting correctness. The ordering within each level is:

```
kickParticlesAtLevel(lev, dt_[lev], accel)   // first half-kick (independent of tOld_/tNew_)
tOld_[lev] = tNew_[lev];                      // update time levels
tNew_[lev] += dt_[lev];                       // update time levels
advanceSingleTimestepAtLevel(lev, ...)         // hydro advance (depends on tOld_/tNew_)
driftParticles(lev, lev, dt_[lev])            // drift (independent of tOld_/tNew_)
```

**Note on caching:** For Option B, the acceleration field comes from the single global Poisson solve. The potential `phi[lev]` is already computed by `calculateGpotAllLevels()` before the subcycling begins. Each call to `computeAccelerationAtLevel(lev)` reads from this existing `phi[lev]` and computes the gradient — it does **not** re-solve Poisson. The cost is just the gradient computation and ghost-cell fill, which is cheap relative to the Poisson solve. However, for a level that subcycles with ratio `nsubsteps`, `computeAccelerationAtLevel` is called `2 * nsubsteps` times (two half-kicks per substep). A caching optimization (see Risk #4) could reduce this to once per level per step.

### Change 3: Remove global particle operations from `evolve()`

**File:** `src/simulation.hpp`, function `evolve()` (around lines 1440-1487)

**Remove these three blocks:**

1. **First global kick** (lines 1440-1447):
   ```cpp
   // REMOVE:
   #if AMREX_SPACEDIM == 3
       if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
           if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
               kickParticlesAllLevels(dt_[0]);
           }
       }
   #endif
   ```

2. **Global drift** (lines 1456-1464):
   ```cpp
   // REMOVE:
   #if AMREX_SPACEDIM == 3
       if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
           if (particleRegister_.HasMassiveParticles()) {
               particleRegister_.driftParticlesAllLevels(dt_[0], finest_level);
           }
       }
   #endif
   ```

3. **Second global kick** (lines 1469-1475):
   ```cpp
   // REMOVE:
   #if AMREX_SPACEDIM == 3
       if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
           if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
               kickParticlesAllLevels(dt_[0]);
           }
       }
   #endif
   ```

**Keep in place (unchanged):**

- `updateParticleProperties` (line 1478)
- `particleMeshInteraction` (line 1481)
- `destroyParticles` (line 1485)

### Change 4: Split `ellipticSolveAllLevels` and reorder in `evolve()`

**File:** `src/simulation.hpp`

**Current code** (`ellipticSolveAllLevels`, line 1920):
```cpp
template <typename problem_t> void AMRSimulation<problem_t>::ellipticSolveAllLevels(const amrex::Real dt)
{
    if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
        if (poissonSupercycleInterval_ > 1) { ... }
        if (istep[0] % poissonSupercycleInterval_ == 0) {
            calculateGpotAllLevels();  // Poisson solve
        }
        gravAccelAllLevels(dt);  // apply gravity to gas
    }
}
```

**Current ordering in `evolve()`:**
```
kickParticlesAllLevels(dt_[0])           // first half-kick (phi from previous step)
timeStepWithSubcycling(...)               // hydro advance
driftParticlesAllLevels(dt_[0])           // drift
ellipticSolveAllLevels(dt_[0])            // Poisson solve (post-hydro, post-drift) + gas gravity
kickParticlesAllLevels(dt_[0])           // second half-kick (phi from current step)
```

**New ordering in `evolve()`:**

The Poisson solve must happen **before** subcycling so that `phi[lev]` is available for per-level kicks inside `timeStepWithSubcycling`. Gas gravity must happen **after** subcycling so it applies to the post-hydro gas state. This requires splitting `ellipticSolveAllLevels` into its two constituents:

```cpp
// Before subcycling: compute gravitational potential for particle kicks
if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
    if (poissonSupercycleInterval_ > 1) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(regrid_int <= 0,
            "Poisson supercycling is only allowed for static meshes!");
    }
    if (istep[0] % poissonSupercycleInterval_ == 0) {
        calculateGpotAllLevels();
    }
}

// Subcycling: hydro advance + per-level particle kicks/drifts
timeStepWithSubcycling(0, cur_time, 1);

// After subcycling: apply gravity to gas using phi computed above
if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
    gravAccelAllLevels(dt_[0]);
}
```

**Time-centering trade-off:** In the current code, the Poisson solve computes `phi` from the post-hydro, post-drift state, and the second half-kick uses this `phi`. This gives time-centered kicks (first half-kick uses old phi, second half-kick uses new phi). In the new code, `phi` is computed from the pre-hydro, pre-drift state, and both half-kicks use this `phi`. The time-centering error is O(dt^2) per step for particle kicks (same order as the leapfrog truncation error). For gas gravity, using pre-hydro `phi` instead of post-hydro `phi` introduces an O(dt) local error, which is comparable to the existing operator-splitting error in the gravitational source term.

The existing `ellipticSolveAllLevels` function can be simplified or removed, since its two operations are now called separately in `evolve()`.

### Change 5: Add `computeAccelerationAtLevel` method declaration and implementation

**File:** `src/simulation.hpp`

**Add declaration** near line 527 (near `kickParticlesAllLevels`):
```cpp
#if AMREX_SPACEDIM == 3
    amrex::MultiFab computeAccelerationAtLevel(int lev);
#endif
```

**Add implementation** near line 1972 (near `kickParticlesAllLevels`):
```cpp
#if AMREX_SPACEDIM == 3
template <typename problem_t>
amrex::MultiFab AMRSimulation<problem_t>::computeAccelerationAtLevel(int lev)
{
    // Build phi_extended with ghost cells and fill boundaries.
    // nghost_acc and nghost_phi may need to be dynamic for subcycling (see Change 8).
    constexpr int nghost_acc = 2;
    constexpr int nghost_phi = nghost_acc + 1;

    amrex::MultiFab phi_extended(boxArray(lev), DistributionMap(lev), 1, nghost_phi);

    amrex::Vector<amrex::BCRec> phiBC(1);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        phiBC[0].setLo(i, BCs_cc_[Physics_Indices<problem_t>::hydroFirstIndex].lo(i));
        phiBC[0].setHi(i, BCs_cc_[Physics_Indices<problem_t>::hydroFirstIndex].hi(i));
    }
    amrex::GpuBndryFuncFab<setFunctorParticleAccel> boundaryFunctor(setFunctorParticleAccel{});

    if (lev == 0) {
        amrex::MultiFab::Copy(phi_extended, phi[lev], 0, 0, 1, 0);
        phi_extended.FillBoundary(geom[lev].periodicity());
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> phiBdryFunct(geom[lev], phiBC, boundaryFunctor);
        phiBdryFunct(phi_extended, 0, 1, phi_extended.nGrowVect(), 0., 0);
    } else {
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> phiBdryFunct(geom[lev], phiBC, boundaryFunctor);
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> phiCoarseBdryFunct(geom[lev - 1], phiBC, boundaryFunctor);
        amrex::FillPatchTwoLevels(phi_extended, 0., {&phi[lev - 1]}, {0.}, {&phi[lev]}, {0.}, 0, 0, 1,
                                   geom[lev - 1], geom[lev], phiCoarseBdryFunct, 0, phiBdryFunct, 0,
                                   refRatio(lev - 1), &amrex::quadratic_interp, phiBC, 0);
    }

    AMREX_ALWAYS_ASSERT(!phi_extended.contains_nan());

    // Compute cell-centered acceleration from potential gradient
    amrex::MultiFab accel_cc(boxArray(lev), DistributionMap(lev), AMREX_SPACEDIM, nghost_acc);
    const auto &phi_arr = phi_extended.const_arrays();
    const auto dx_inv = geom[lev].InvCellSizeArray();
    auto accel_arr = accel_cc.arrays();

    amrex::ParallelFor(accel_cc, amrex::IntVect{nghost_acc},
        [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
            accel_arr[bx](i, j, k, 0) = -0.5 * dx_inv[0] * (phi_arr[bx](i + 1, j, k) - phi_arr[bx](i - 1, j, k));
            accel_arr[bx](i, j, k, 1) = -0.5 * dx_inv[1] * (phi_arr[bx](i, j + 1, k) - phi_arr[bx](i, j - 1, k));
            accel_arr[bx](i, j, k, 2) = -0.5 * dx_inv[2] * (phi_arr[bx](i, j, k + 1) - phi_arr[bx](i, j, k - 1));
        });
    amrex::Gpu::streamSynchronize();

    return accel_cc;
}
#endif
```

Then refactor `kickParticlesAllLevels` to use it:
```cpp
template <typename problem_t> void AMRSimulation<problem_t>::kickParticlesAllLevels(const amrex::Real dt)
{
    if (!particleRegister_.HasMassiveParticles()) { return; }
    for (int lev = 0; lev <= finest_level; ++lev) {
        auto accel_cc = computeAccelerationAtLevel(lev);
        particleRegister_.kickParticlesAtLevel(lev, dt, accel_cc);
    }
}
```

### Change 6: Handle the Poisson supercycling interval check

**File:** `src/simulation.hpp`

The current `ellipticSolveAllLevels` (line 1920) gates the Poisson solve on `istep[0] % poissonSupercycleInterval_`. When we split the solve from the gas gravity application, we need the same gating for `calculateGpotAllLevels`.

Move the supercycling check into `evolve()`:
```cpp
// Before subcycling:
if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
    if (poissonSupercycleInterval_ > 1) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(regrid_int <= 0,
            "Poisson supercycling is only allowed for static meshes!");
    }
    if (istep[0] % poissonSupercycleInterval_ == 0) {
        calculateGpotAllLevels();
    }
}

timeStepWithSubcycling(0, cur_time, 1);

// After subcycling:
if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
    gravAccelAllLevels(dt_[0]);
}
```

And simplify `ellipticSolveAllLevels` accordingly (or remove it and inline the calls).

### Change 7: Address existing TODOs about AMR subcycling

**File:** `src/simulation.hpp`, lines 1480-1485

These existing TODOs:
```cpp
// TODO(cch): Need to take care of AMR subcycling
particleMeshInteraction(cur_time, dt_[0]);
// TODO(cch): Need to take care of AMR subcycling
particleRegister_.destroyParticles(0, cur_time, dt_[0]);
```

For now, `particleMeshInteraction` and `destroyParticles` can remain at the finest level after all subcycling completes (current behavior). Per-level particle-mesh interaction is a separate enhancement that requires careful coordination with flux registration. The drift-and-kick changes described here do not depend on moving these.

### Change 8: Acceleration field ghost cells for subcycling (Nyx-inspired)

**File:** `src/simulation.hpp`, function `computeAccelerationAtLevel`

**Problem:** When subcycling is active, particles at fine levels drift by `dt_[lev+1]` per substep. Over `nsubsteps[lev+1]` substeps, a particle can drift by up to `nsubsteps * v_max * dt_[lev+1]` cells relative to its starting position. The acceleration field needs enough ghost cells for CIC interpolation to remain valid at all drift positions during the substeps.

**Nyx's solution** (`Nyx_advance.cpp:99-116`):
```
ghost_width  = ncycle + stencil_deposition_width
grav_n_grow  = ghost_width + stencil_interpolation_width
```
where `ncycle` is the subcycling ratio at the current level and `stencil_deposition_width = stencil_interpolation_width = 1` for CIC.

**Quokka's current code** (`simulation.hpp:1987`): The acceleration MultiFab is built with `nghost_acc = 2` ghost cells, which is sufficient for CIC interpolation when there is no subcycling (particle drifts at most 1 cell). For subcycling, this must increase.

**Proposed change:** In `computeAccelerationAtLevel`, compute the required ghost cells dynamically:

```cpp
// Number of ghost cells for the acceleration field.
// With no subcycling, particles drift at most 1 cell, so nghost_acc = 2 suffices for CIC.
// With subcycling, particles can drift up to nsubsteps cells over all substeps.
// Use max(2, 1 + nsubsteps[lev]) to ensure CIC remains valid.
const int nghost_acc_local = (do_subcycle != 0 && lev < static_cast<int>(nsubsteps.size()) - 1)
    ? (1 + nsubsteps[lev + 1])
    : 2;
const int nghost_phi_local = nghost_acc_local + 1;
```

**Note:** When `do_subcycle == 0` (the current default), `nghost_acc_local = 2` and behavior is unchanged. When subcycling is enabled, the ghost cell count increases proportionally to the subcycling ratio.

### Change 9: Post-drift grid assignment check (Nyx-inspired)

**File:** `src/particles/PhysicsParticles.hpp`, method `driftParticles`

**Problem:** After drifting, particles at fine levels (`lev > 0`) may have moved outside the valid region of their current grid tile. In Nyx, this is handled by re-checking each particle's grid assignment with a `where_width` parameter that shrinks as iterations progress:

```
where_width = ghost_width + (1 - iteration) - 1
```

This ensures that:
1. On iteration 1 (first substep), particles could be up to `ghost_width` cells out of their grid, so a generous check is needed.
2. On later iterations, particles have already been redistributed, so they can be at most 1 cell out.

**Quokka's current code** (`simulation.hpp:2253-2264`): Redistribution after each substep already uses `nGrow = iteration`, which follows the same pattern as Nyx. However, the `driftParticles` method itself does not check whether a particle has moved out of valid grid territory.

**Proposed change:** Add an optional `where_width` parameter to `driftParticles` that, when subcycling is active (`lev > 0 && do_subcycle != 0`), performs a grid assignment check after the drift. Particles that have moved outside the valid `where_width` cells of their grid should be flagged for removal (if they are ghost particles) or left for the subsequent `Redistribute` call to handle (if they are active particles).

For the initial implementation without ghost/virtual particles (see Change 10), this check is less critical because `Redistribute` already handles cross-grid movement. However, it is important for correctness when ghost particles are eventually introduced, because ghost particles that drift too far must be invalidated to prevent double-counting.

**Implementation sketch** in `PhysicsParticles.hpp`:
```cpp
void driftParticles(int lev_min, int lev_max, amrex::Real dt, int where_width = 0) const
{
    // ... existing drift kernel ...

    // Post-drift grid assignment check for subcycling levels
    if (where_width > 0 && lev > 0) {
        // Check each particle against grid assignment with where_width tolerance
        // Mark out-of-bounds ghost particles for removal (id = -1)
        // Active particles will be handled by the subsequent Redistribute call
    }
}
```

And in `timeStepWithSubcycling`, compute `where_width` following Nyx's formula:
```cpp
const int ncycle = (lev < static_cast<int>(nsubsteps.size()) - 1) ? nsubsteps[lev + 1] : 1;
const int ghost_width = ncycle + 1;  // stencil_deposition_width = 1
const int where_width = ghost_width + (1 - iteration) - 1;
```

### Change 10 (Future): Ghost and Virtual Particles

This is a **future enhancement** not required for the initial implementation, but documented here for completeness.

Nyx uses ghost and virtual particles to ensure consistent cross-level particle evolution during subcycling:

- **Virtual particles**: Fine-level particles projected to coarse levels. Without them, the coarse-level gravity solve "misses" the mass of fine-level particles, causing errors in the gravitational potential near coarse-fine boundaries.

- **Ghost particles**: Coarse-level particles projected to fine levels. Without them, fine-level particles near coarse-fine boundaries don't feel the gravitational influence of nearby coarse-level particles during their substeps. This creates asymmetric forces at the boundary.

**When to add these:** Ghost/virtual particles become important when:
1. Subcycling is enabled (`do_subcycle != 0`).
2. Particles are near coarse-fine AMR boundaries.
3. Gravitational accuracy at coarse-fine boundaries matters for the science problem.

**For Option B without ghost/virtual particles:** The dominant error is an O(dt) time-centering mismatch for particles near coarse-fine boundaries. These particles feel the coarse-level potential for their full coarse timestep, while gas on the fine level evolves with the fine timestep. This is acceptable for many cosmological problems where the gravitational potential is smooth and the subcycling ratio is small (2-4).

### Change 11 (Future): Time-centered kicks with phi_prev/phi_new

This is a **future enhancement** to improve time-centering accuracy beyond what Option B provides with a single phi.

The current code uses phi_prev (from the previous step's Poisson solve) for the first half-kick and phi_new (from the current step's Poisson solve) for the second half-kick, giving time-centered kicks. Option B with a single Poisson solve before subcycling uses the same phi for both half-kicks, introducing an O(dt^2) time-centering error per step.

To restore time-centering while keeping per-level KDK inside subcycling, one could:

1. Use phi_prev for the first half-kick (already available from the previous step).
2. Inside subcycling, do both half-kicks with phi_prev (synchronous KDK).
3. After subcycling, compute phi_new via `calculateGpotAllLevels()`.
4. Apply a correction kick to adjust from phi_prev to phi_new:
   ```
   vel += 0.5 * dt_[lev] * (accel_new - accel_prev)
   ```
   This corrects the second half-kick at each level from phi_prev to phi_new.

For fine levels with `nsubsteps` substeps, the total correction would need to account for the cumulative effect of multiple half-kicks using phi_prev instead of phi_new. This requires either storing the old acceleration field per level or recomputing it from the stored phi_prev.

This enhancement can be implemented after the initial per-level KDK framework is working and tested.

---

## Ordering Summary

### Before (current)

```
evolve():
    kickParticlesAllLevels(dt_[0])            // first half-kick (phi from previous step)
    timeStepWithSubcycling(0, t, 1)           // hydro only, no particles
    driftParticlesAllLevels(dt_[0])           // global drift
    ellipticSolveAllLevels(dt_[0])            // Poisson solve (post-hydro) + gas gravity
    kickParticlesAllLevels(dt_[0])            // second half-kick (phi from current step)
    updateParticleProperties / particleMeshInteraction / destroyParticles
```

Note: `kickParticlesAllLevels(dt_[0])` performs a half-kick (`vel += 0.5 * dt * accel`) because `kickParticles` internally applies the 0.5 factor. Two calls give the full kick for the KDK cycle.

### After (Option B)

```
evolve():
    calculateGpotAllLevels()                  // Poisson solve: phi from current (pre-hydro) state

    timeStepWithSubcycling(0, t, 1):          // per-level advance
        kickParticlesAtLevel(lev, dt_[lev], accel_lev)   // first half-kick (0.5 * dt_[lev] * accel)
        tOld_[lev] = tNew_[lev]                          // update time levels
        tNew_[lev] += dt_[lev]                           // update time levels
        advanceSingleTimestepAtLevel(lev)                 // hydro advance
        driftParticles(lev, lev, dt_[lev])               // per-level drift
        // recurse into finer levels (nsubsteps[lev+1] sub-steps)
        kickParticlesAtLevel(lev, dt_[lev], accel_lev)   // second half-kick (0.5 * dt_[lev] * accel)
        reflux / AverageDown / FixupState
        redistribute particles at level lev (nGrow = iteration for lev > 0)

    gravAccelAllLevels(dt_[0])                // gas gravity (using same phi as particle kicks)
    updateParticleProperties / particleMeshInteraction / destroyParticles
```

Note: `kickParticlesAtLevel(lev, dt_[lev], accel)` performs a half-kick because `kickParticles` internally applies 0.5. The full `dt_[lev]` is passed, not `dt_[lev]/2`.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/simulation.hpp` | Add `computeAccelerationAtLevel(lev)` declaration and implementation; refactor `kickParticlesAllLevels` to use it; insert per-level drift/kick in `timeStepWithSubcycling`; remove global drift/kick from `evolve()`; split `ellipticSolveAllLevels` into separate `calculateGpotAllLevels` + `gravAccelAllLevels` calls in `evolve()` |
| `src/particles/PhysicsParticles.hpp` | Add optional `where_width` parameter to `driftParticles` for post-drift grid assignment check (initially unused; needed when ghost particles are added) |

---

## Risks and Open Questions

1. **Poisson solve timing and time-centering:** Moving `calculateGpotAllLevels()` before subcycling means `phi` is computed from the **pre-hydro, pre-drift** state. In the current code, `phi` is computed from the **post-hydro, post-drift** state, and the second half-kick uses this more accurate `phi`. The new scheme uses the same `phi` for both half-kicks, introducing an O(dt^2) local error per step for particle kicks — the same order as the leapfrog truncation error. For gas gravity, using pre-hydro `phi` instead of post-hydro `phi` introduces an O(dt) local error, comparable to the operator-splitting error already present in the gravitational source term. This is a known trade-off of Option B's single Poisson solve. See Change 11 for a future enhancement that restores time-centering.

2. **Gas gravity accuracy:** Currently `gravAccelAllLevels(dt_[0])` applies the gravitational source term to gas using `phi` computed from post-hydro density. In the new code, gas gravity uses `phi` computed from pre-hydro density (the same `phi` used for particle kicks). This is a regression in gas gravity accuracy. If this proves problematic, one could compute `phi` twice (once before subcycling for particle kicks, once after for gas gravity), but this doubles the Poisson solve cost and contradicts the single-solve premise of Option B. Alternatively, the future enhancement in Change 11 would allow computing `phi` after subcycling for both the correction kick and gas gravity, restoring gas gravity accuracy.

3. **Poisson solve mass deposition with pre-drift positions:** In the current code, the Poisson solve uses post-drift particle positions for mass deposition (particles are drifted before the Poisson solve). In the new code, the Poisson solve uses pre-drift positions (particles are drifted inside subcycling, after the Poisson solve). This means the gravitational potential is computed for where the particles currently are, not where they will be after the drift. The O(v*dt) position lag is comparable to the lag in the current code's first half-kick, which uses `phi` from the previous step's post-drift positions.

4. **Subcycling correctness when `do_subcycle == 1`:** Currently the code aborts if subcycling is enabled with self-gravity. This change does not enable subcycling with gravity — it merely makes the particle integration subcycling-ready. The assertion in `calculateGpotAllLevels` (line 1711) should remain until a per-substep Poisson solve is implemented (Option A).

5. **Acceleration field recomputation:** Each call to `computeAccelerationAtLevel` recomputes the acceleration field from `phi[lev]`. For Option B (single Poisson solve), this is redundant if called multiple times per step for the same level. A caching optimization could store `accel_cc` per level and reuse it. This is optional for correctness but important for performance. Consider adding a `amrex::Vector<amrex::MultiFab> accel_cache_` member that is invalidated when `calculateGpotAllLevels` is called, and populated lazily by `computeAccelerationAtLevel`.

6. **Particle redistribution:** Already handled per-level in `timeStepWithSubcycling` (lines 2253-2264) with the correct `nGrow = iteration` pattern matching Nyx. No change needed.

7. **Particle creation/destruction (particleMeshInteraction, destroyParticles):** Currently at finest level only. These remain at finest level after all subcycling. Per-level creation/destruction is a future enhancement.

8. **Ghost cells for acceleration field under subcycling:** Change 8 addresses this. The current `nghost_acc = 2` is correct for no subcycling but must increase when subcycling is enabled. The Nyx formula `grav_n_grow = ncycle + stencil_interpolation_width` provides the correct count.

9. **Post-drift grid assignment:** Change 9 documents the Nyx-inspired `where_width` check. For the initial implementation without ghost particles, `Redistribute(nGrow=iteration)` suffices. When ghost particles are added, the `where_width` check becomes essential for invalidating ghost particles that have drifted out of their valid region.

10. **Regression with `do_subcycle == 0`:** When subcycling is disabled, `dt_[lev] == dt_[0]` for all levels, and the per-level KDK with `dt_[lev]` should produce exactly the same result as the global KDK with `dt_[0]`. However, the Poisson solve timing change (before vs. after subcycling) means `phi` is computed from a different state, which will cause bit-for-bit differences even with `do_subcycle == 0`. To verify correctness, compare against a run with the old code on the same problem and check that the differences are O(dt^2) (consistent with the time-centering error documented in Risk #1).

---

## Testing Strategy

1. **Regression test with `do_subcycle == 0`:** Run an existing gravity + particles test (e.g., `ParticleSink`, `ParticleSF`, or `DiskGalaxy`) and compare results before and after the change. Note that results will NOT be bit-for-bit identical due to the Poisson solve timing change (see Risk #10). Verify that differences are small and consistent with O(dt^2) time-centering error.

2. **Subcycling test:** Enable `do_subcycle = 1` with a 2-level AMR hierarchy and particles. Verify that fine-level particles take smaller steps. Compare trajectory accuracy against a high-resolution uniform-grid run.

3. **Conservation check:** Verify that total momentum (gas + particles) is conserved to the expected order with the new scheme. The leapfrog KDK scheme should still be second-order in time.

4. **Performance:** Profile `computeAccelerationAtLevel` overhead. If called multiple times per level per step (two half-kicks per substep, with subcycling), consider implementing the caching optimization (Risk #5).

5. **Ghost cell validation:** When subcycling is enabled, verify that CIC interpolation in `kickParticles` produces valid results by checking that particles remain within the ghost cell region of the acceleration MultiFab. This can be done by asserting that `kickParticles` does not access out-of-bounds cells.