# IMEX Butcher Tableau Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the radiation IMEX scheme so that it is parameterized by a Butcher tableau defined as constexpr, and uses explicit temporary state storage (`state_tmp1_cc`) instead of the gas_update_factor "half-step backward trick."

**Architecture:** The current IMEX PD-ARS scheme hardcodes two coefficients (`IMEX_a22`, `IMEX_a32`) in `radiation_system.hpp` and threads a `stage` parameter through source-term functions to compute derived quantities (dt, gas_update_factor). The new design defines the full Butcher tableau as named constexpr entries at the top of `QuokkaSimulation.hpp`, stores intermediate IMEX stages in a temporary MultiFab (`state_tmp1_cc`), and passes computed `dt_implicit` / `gas_update_factor` explicitly to source-term functions. This eliminates the implicit coupling between tableau entries and source-term internals.

**Tech Stack:** C++20, AMReX, GPU-compatible (CUDA/HIP)

**Behavioral notes:**
- For **single-group** radiation: the refactor is algebraically equivalent. In single-group `AddSourceTermsSingleGroup`, `gas_update_factor` is applied only at the final storage step (lines 546-553 of `source_terms_single_group.hpp`), not during the Newton-Raphson iteration. The Shu-Osher combination with full gas update at stage 2 followed by `(1-alpha)*gas_n + alpha*gas_stage2` at stage 3 produces the same starting point for stage 3's implicit solve.
- For **multi-group** radiation: the behavior intentionally changes. In multi-group `AddSourceTermsMultiGroup`, `gas_update_factor` enters the `UpdateFlux` function (line 579 of `source_terms_multi_group.hpp`), affecting momentum during the work-term convergence loop. Setting `gas_update_factor=1.0` implements the mathematically correct IMEX scheme; the current `gas_update_factor=0.5` was a numerical approximation. This is the intended design change (removing the "half-step backward trick").

---

## Mathematical Background

The 3-stage IMEX PD-ARS scheme has explicit tableau Aex and implicit tableau Aim:

```
Explicit (Aex):          Implicit (Aim):
0 | 0   0   0            0 | 0   0   0
1 | 1   0   0            1 | 0   1   0
1 | 1/2 1/2 0            1 | 0  1/2 1/2
  |---------                |----------
  | 1/2 1/2 0              | 0  1/2 1/2
```

The scheme is **stiffly accurate** (b = last row of A). Stages:

- **Stage 1**: U^(1) = U^n (trivial)
- **Stage 2**: U^(2) = U^n + dt * Aex_21 * s(U^(1)) + dt * Aim_22 * g(U^(2))
- **Stage 3**: U^(3) = U^n + dt * [Aex_31*s(U^(1)) + Aex_32*s(U^(2))] + dt * [Aim_32*g(U^(2)) + Aim_33*g(U^(3))]

where s = explicit flux divergence (radiation transport), g = implicit source (matter-radiation coupling).

**Shu-Osher form for Stage 3** (avoids separately storing g(U^(2))):

Using U^(2) = U^n + dt*Aex_21*s(U^(1)) + dt*Aim_22*g(U^(2)), we can derive:

```
let alpha = Aim_32 / Aim_22
U^(3)* = (1-alpha)*U^n + alpha*U^(2) + dt*(Aex_31 - alpha*Aex_21)*s(U^(1)) + dt*Aex_32*s(U^(2))
```

Then solve implicitly: U^(3) = U^(3)\* + dt * Aim_33 * g(U^(3))

For PD-ARS: alpha = 0.5/1.0 = 0.5, so:
```
U^(3)* = 0.5*U^n + 0.5*U^(2) + 0*s(U^(1)) + 0.5*dt*s(U^(2))
```

This matches the current `AddFluxesRK2` formula (line 813 of radiation_system.hpp).

**Critical subtlety:** `AddFluxesRK2` only operates on radiation hyperbolic variables (indices `nstartHyperbolic_` to `nstartHyperbolic_ + ncompHyperbolic_`). Gas variables (momentum, energy) are modified only by `AddSourceTerms` and must have the Shu-Osher combination applied **separately** after `AddFluxesRK2`. This is done with `MultiFab::LinComb` on the non-hyperbolic component range.

**Equivalence proof for gas (single-group):** Let gas_n = hydro-updated gas before radiation coupling. After stage 2 with gas_update_factor=1: gas_stage2 = gas_n + delta_g. The Shu-Osher predictor gives gas_pred = (1-alpha)*gas_n + alpha*gas_stage2 = gas_n + alpha*delta_g. In the current code: gas after stage 1 = gas_n + alpha*delta_g (since gas_update_factor=alpha=IMEX_a32). Both produce the same stage 3 starting point.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/QuokkaSimulation.hpp` | Modify | Add Butcher tableau constexpr, allocate `state_tmp1_cc`, rewrite `subcycleRadiationAtLevel`, update `advanceRadiationForwardEuler`, update `advanceRadiationMidpointRK2`, add Shu-Osher gas combination |
| `src/radiation/radiation_system.hpp` | Modify | Remove old `IMEX_a22`/`IMEX_a32` constants, update `AddFluxesRK2` signature to accept tableau coefficients as params |
| `src/radiation/source_terms_single_group.hpp` | Modify | Replace `stage` parameter with explicit `dt_implicit` + `gas_update_factor` params |
| `src/radiation/source_terms_multi_group.hpp` | Modify | Same changes as single_group |

---

### Task 1: Define Butcher Tableau Constants

**Files:**
- Modify: `src/QuokkaSimulation.hpp` (near line 177, with other `static constexpr`)
- Modify: `src/radiation/radiation_system.hpp` (lines 50-56, remove old constants)

- [ ] **Step 1.1: Add Butcher tableau constexpr to QuokkaSimulation.hpp**

Add after the existing `static constexpr` block (around line 183, after `numDustVars_`):

```cpp
// IMEX PD-ARS Butcher tableau
// Explicit tableau (strictly lower triangular): Aex_ij, i > j
static constexpr double IMEX_Aex_21 = 1.0;
static constexpr double IMEX_Aex_31 = 0.5;
static constexpr double IMEX_Aex_32 = 0.5;
// Implicit tableau (lower triangular with diagonal): Aim_ij, i >= j
static constexpr double IMEX_Aim_22 = 1.0;
static constexpr double IMEX_Aim_32 = 0.5;
static constexpr double IMEX_Aim_33 = 0.5;
// Derived coefficient for Shu-Osher form of stage 3
// Guard: IMEX_Aim_22 must be > 0 for the Shu-Osher form to be valid
static_assert(IMEX_Aim_22 > 0.0, "IMEX_Aim_22 must be > 0 for the IMEX PD-ARS scheme");
static constexpr double IMEX_alpha = IMEX_Aim_32 / IMEX_Aim_22; // = 0.5
```

- [ ] **Step 1.2: Remove old constants from radiation_system.hpp**

Delete lines 50-56 (the `IMEX_a22`, `IMEX_a32` definitions and comments) from `src/radiation/radiation_system.hpp`.

- [ ] **Step 1.3: Verify build fails**

Run: `cd $(pwd)/tests && JOB=RadMarshak make b 2>&1 | tail -20`
Expected: Compilation errors referencing `IMEX_a22`, `IMEX_a32` in source_terms and AddFluxesRK2.

- [ ] **Step 1.4: Commit**

```bash
git add src/QuokkaSimulation.hpp src/radiation/radiation_system.hpp
git commit -m "refactor: define IMEX Butcher tableau as named constexpr entries

Replace IMEX_a22/IMEX_a32 with full Butcher tableau: IMEX_Aex_21, IMEX_Aex_31,
IMEX_Aex_32, IMEX_Aim_22, IMEX_Aim_32, IMEX_Aim_33, IMEX_alpha.
Build is intentionally broken at this commit."
```

---

### Task 2: Update Source Term Functions to Accept Explicit Parameters

**Files:**
- Modify: `src/radiation/source_terms_single_group.hpp` (lines 10-19, 88-91)
- Modify: `src/radiation/source_terms_multi_group.hpp` (lines 589-600, 688-691)
- Modify: `src/radiation/radiation_system.hpp` (lines 293-299, declarations)

The goal is to replace the `stage` integer parameter with explicit `dt_implicit` and `gas_update_factor` parameters. This removes the source-term functions' dependency on global IMEX coefficients.

- [ ] **Step 2.1: Modify AddSourceTermsSingleGroup signature and body**

In `src/radiation/source_terms_single_group.hpp`:

Change signature (lines 10-12) from:
```cpp
void RadSystem<problem_t>::AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, Real dt_radiation,
                                                     const int stage, double dustGasCoeff, double tol_h, double /*tol_rel_h*/, double /*tempFloor*/,
                                                     int *p_iteration_counter, int *p_iteration_failure_counter)
```
to:
```cpp
void RadSystem<problem_t>::AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, Real dt_implicit,
                                                     double gas_update_factor_in, double dustGasCoeff, double tol_h, double /*tol_rel_h*/,
                                                     double /*tempFloor*/, int *p_iteration_counter, int *p_iteration_failure_counter)
```

Replace the stage-dependent dt computation (lines 14-19) with:
```cpp
arrayconst_t &consPrev = consVar; // make read-only
array_t &consNew = consVar;
auto dt = dt_implicit;
```

Replace the gas_update_factor computation (lines 88-91) with:
```cpp
Real gas_update_factor = gas_update_factor_in;
```

- [ ] **Step 2.2: Update AddSourceTermsSingleGroup declaration in radiation_system.hpp**

Change the declaration at line 297-299 from:
```cpp
static void AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt, int stage,
                                      double dustGasCoeff, double tol_h, double tol_rel_h, double tempFloor, int *p_iteration_counter,
                                      int *p_iteration_failure_counter);
```
to:
```cpp
static void AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_implicit,
                                      double gas_update_factor, double dustGasCoeff, double tol_h, double tol_rel_h, double tempFloor,
                                      int *p_iteration_counter, int *p_iteration_failure_counter);
```

- [ ] **Step 2.3: Modify AddSourceTermsMultiGroup signature and body**

In `src/radiation/source_terms_multi_group.hpp`, apply the same changes:

Change signature (lines 589-591) — replace `amrex::Real dt_radiation, const int stage` with `amrex::Real dt_implicit, double gas_update_factor_in`.

Replace stage-dependent dt computation (lines 596-600) with:
```cpp
arrayconst_t &consPrev = consVar;
array_t &consNew = consVar;
auto dt = dt_implicit;
```

Replace gas_update_factor computation (lines 688-691) with:
```cpp
amrex::Real gas_update_factor = gas_update_factor_in;
```

- [ ] **Step 2.4: Update AddSourceTermsMultiGroup declaration in radiation_system.hpp**

Change the declaration at line 293-295 from:
```cpp
static void AddSourceTermsMultiGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt, int stage,
                                     double dustGasCoeff, double tol_h, double tol_rel_h, double tempFloor, int *p_iteration_counter,
                                     int *p_iteration_failure_counter);
```
to:
```cpp
static void AddSourceTermsMultiGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_implicit,
                                     double gas_update_factor, double dustGasCoeff, double tol_h, double tol_rel_h, double tempFloor,
                                     int *p_iteration_counter, int *p_iteration_failure_counter);
```

- [ ] **Step 2.5: Commit**

```bash
git add src/radiation/source_terms_single_group.hpp src/radiation/source_terms_multi_group.hpp src/radiation/radiation_system.hpp
git commit -m "refactor: source term functions accept dt_implicit and gas_update_factor as params

Replace stage-based internal logic with explicit parameters. Callers now
compute the correct dt and gas_update_factor from the Butcher tableau."
```

---

### Task 3: Update AddFluxesRK2 to Accept Tableau Coefficients

**Files:**
- Modify: `src/radiation/radiation_system.hpp` (lines 768-814, and forward declaration)

- [ ] **Step 3.1: Add tableau parameters to AddFluxesRK2 signature**

Change the signature (lines 768-772) to accept the Shu-Osher coefficients:

```cpp
template <typename problem_t>
void RadSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1,
                                        amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
                                        amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
                                        amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArrayOld*/,
                                        amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/,
                                        const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
                                        amrex::Box const &indexRange, const int /*nvars*/,
                                        const double alpha, const double Aex_s1_coeff, const double Aex_s2_coeff)
```

Where:
- `alpha` = Aim_32 / Aim_22 (Shu-Osher weight for U^(2))
- `Aex_s1_coeff` = Aex_31 - alpha * Aex_21 (flux coefficient for s(U^(1)))
- `Aex_s2_coeff` = Aex_32 (flux coefficient for s(U^(2)))

- [ ] **Step 3.2: Update the kernel formula and comment**

Replace the comment at line 797 and the formula at line 813.

Old comment:
```cpp
// y^n+1 = (1 - a32) y^n + a32 y^(2) + dt * (0.5 - a32) * s(y^n) + dt * 0.5 * s(y^(2)) + dt * (1 - a32) * f(y^n+1)
```
New comment:
```cpp
// Shu-Osher form: y^(3)* = (1-alpha)*y^n + alpha*y^(2) + dt*Aex_s1_coeff*s(y^n) + dt*Aex_s2_coeff*s(y^(2))
// where alpha = Aim_32/Aim_22, Aex_s1_coeff = Aex_31 - alpha*Aex_21, Aex_s2_coeff = Aex_32
// The implicit term dt*Aim_33*g(y^(3)) is handled separately in subcycleRadiationAtLevel.
```

Old formula:
```cpp
cons_new[n] = (1.0 - IMEX_a32) * U_0 + IMEX_a32 * U_1 + ((0.5 - IMEX_a32) * (AMREX_D_TERM(FxU_0, +FyU_0, +FzU_0))) +
              (0.5 * (AMREX_D_TERM(FxU_1, +FyU_1, +FzU_1)));
```
New formula:
```cpp
cons_new[n] = (1.0 - alpha) * U_0 + alpha * U_1 + (Aex_s1_coeff * (AMREX_D_TERM(FxU_0, +FyU_0, +FzU_0))) +
              (Aex_s2_coeff * (AMREX_D_TERM(FxU_1, +FyU_1, +FzU_1)));
```

- [ ] **Step 3.3: Update the forward declaration in radiation_system.hpp**

Find the declaration of `AddFluxesRK2` (around line 277) and add the three new parameters.

- [ ] **Step 3.4: Commit**

```bash
git add src/radiation/radiation_system.hpp
git commit -m "refactor: AddFluxesRK2 accepts Shu-Osher tableau coefficients as params"
```

---

### Task 4: Add Temporary State Storage

**Files:**
- Modify: `src/QuokkaSimulation.hpp` (in `subcycleRadiationAtLevel`)

- [ ] **Step 4.1: Allocate state_tmp1_cc locally in subcycleRadiationAtLevel**

Rather than adding a class member (which would consume memory at all times), allocate the temporary MultiFab at the top of `subcycleRadiationAtLevel`, just before the substep loop. Add after line 2929 (`AMREX_ALWAYS_ASSERT(dt_radiation > 0.0);`):

```cpp
// Temporary state for IMEX stage 2 result (avoids gas_update_factor trick)
amrex::MultiFab state_tmp1_cc(grids[lev], dmap[lev], state_new_cc_[lev].nComp(), nghost_cc_);
```

- [ ] **Step 4.2: Commit**

```bash
git add src/QuokkaSimulation.hpp
git commit -m "refactor: allocate temporary MultiFab for IMEX stage storage"
```

---

### Task 5: Rewrite subcycleRadiationAtLevel and Radiation Advance Functions

**Files:**
- Modify: `src/QuokkaSimulation.hpp` (lines 2930-3213)

This is the main structural change. The substep loop currently has:
1. Forward Euler → state_new (explicit stage 2)
2. AddSourceTerms(stage=1) → modifies state_new in place (implicit stage 2, partial gas update)
3. Midpoint RK2 → state_new (explicit stage 3, using partially-updated state_new as U^(2))
4. AddSourceTerms(stage=2) → modifies state_new in place (implicit stage 3, remaining gas update)

The new flow is:
1. Copy state_new → state_tmp1 (to get hydro-updated gas)
2. Forward Euler → state_tmp1 (overwrites radiation vars from state_old, gas unchanged)
3. AddSourceTerms(dt=Aim_22*dt, gas_factor=1.0) → modifies state_tmp1 in place (implicit stage 2, FULL update)
4. Midpoint RK2 → state_new (explicit stage 3 for radiation, uses state_tmp1 as U^(2))
5. **LinComb for gas**: state_new_gas = (1-alpha)*state_new_gas + alpha*state_tmp1_gas
6. AddSourceTerms(dt=Aim_33*dt, gas_factor=1.0) → modifies state_new in place (implicit stage 3, FULL update)

**Why LinComb is needed (Step 5):** `AddFluxesRK2` only operates on radiation hyperbolic variables (indices `nstartHyperbolic_` to `nstartHyperbolic_ + ncompHyperbolic_`). Gas variables (momentum, energy, internal energy) are NOT touched by `AddFluxesRK2`. The Shu-Osher combination must be explicitly applied to gas variables. For variables not modified by `AddSourceTerms` (e.g., density), state_new and state_tmp1 hold the same value, so the combination is a no-op.

- [ ] **Step 5.1: Modify advanceRadiationForwardEuler to write to a specified output MultiFab**

Currently `advanceRadiationForwardEuler` reads from `state_old_cc_[lev]` and writes to `state_new_cc_[lev]`. Add an output MultiFab reference parameter.

New signature:
```cpp
void QuokkaSimulation<problem_t>::advanceRadiationForwardEuler(int lev, amrex::Real time, amrex::Real dt_radiation,
                                                                int const /*iter_count*/, int const /*nsubsteps*/,
                                                                amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine,
                                                                amrex::MultiFab &state_out)
```

Replace all `state_new_cc_[lev]` references inside this function with `state_out`:
- Line 3121: `state_new_cc_[lev].boxArray()` → `state_out.boxArray()`
- Line 3123: `state_new_cc_[lev].nComp()` → `state_out.nComp()`
- Line 3135: `for (amrex::MFIter iter(state_new_cc_[lev])` → `for (amrex::MFIter iter(state_out)`
- Line 3138: `state_new_cc_[lev].array(iter)` → `state_out.array(iter)`
- Line 3148: `state_new_cc_[lev].nComp()` → `state_out.nComp()`

- [ ] **Step 5.2: Modify advanceRadiationMidpointRK2 to read intermediate from a specified MultiFab**

Currently it reads U^(2) from `state_new_cc_[lev]`. Now it should read from `state_tmp1_cc`. Add a non-const MultiFab reference parameter (non-const because `fillBoundaryConditions` needs to write ghost cells):

New signature:
```cpp
void QuokkaSimulation<problem_t>::advanceRadiationMidpointRK2(int lev, amrex::Real time, amrex::Real dt_radiation,
                                                               int const /*iter_count*/, int const /*nsubsteps*/,
                                                               amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine,
                                                               amrex::MultiFab &state_inter)
```

Changes inside the function:
- Line 3181: `fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], ...)` → `fillBoundaryConditions(state_inter, state_inter, ...)`
- Line 3188: `auto const &stateInter_cc = state_new_cc_[lev].const_array(iter);` → `auto const &stateInter_cc = state_inter.const_array(iter);`
- Line 3191: `computeRadiationFluxes(stateInter_cc, ...)` stays the same (already uses the local variable)

Pass the Shu-Osher coefficients to `AddFluxesRK2`:
```cpp
RadSystem<problem_t>::AddFluxesRK2(
    stateNew_cc, stateOld_cc, stateInter_cc,
    {AMREX_D_DECL(fluxArraysOld[0].array(), fluxArraysOld[1].array(), fluxArraysOld[2].array())},
    {AMREX_D_DECL(fluxArrays[0].array(), fluxArrays[1].array(), fluxArrays[2].array())},
    {AMREX_D_DECL(fluxDiffusiveArraysOld[0].const_array(), fluxDiffusiveArraysOld[1].const_array(), fluxDiffusiveArraysOld[2].const_array())},
    {AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(), fluxDiffusiveArrays[1].const_array(), fluxDiffusiveArrays[2].const_array())},
    dt_radiation, dx, indexRange, ncompHyperbolic_,
    IMEX_alpha, IMEX_Aex_31 - IMEX_alpha * IMEX_Aex_21, IMEX_Aex_32);
```

- [ ] **Step 5.3: Rewrite the substep loop body in subcycleRadiationAtLevel**

Replace lines 2942-3043 with:

```cpp
// We use the IMEX PD-ARS scheme to evolve the radiation subsystem and radiation-matter coupling.
// The scheme is parameterized by Butcher tableau constexpr defined at the top of this file.

// === IMEX Stage 2 ===
// Initialize state_tmp1 with hydro-updated state (gas variables from state_new_cc_)
// PredictStep will overwrite radiation hyperbolic vars from state_old_cc_
amrex::MultiFab::Copy(state_tmp1_cc, state_new_cc_[lev], 0, 0, state_tmp1_cc.nComp(), 0);

// Explicit: state_tmp1_rad = state_old_rad + dt * Aex_21 * s(state_old_rad)
advanceRadiationForwardEuler(lev, time_subcycle, dt_radiation * IMEX_Aex_21, i, nsubSteps, fr_as_crse, fr_as_fine, state_tmp1_cc);

// failure counter for: matter-radiation coupling, dust temperature, outer iteration
amrex::Gpu::Buffer<int> iteration_failure_counter({0, 0, 0});
// iteration counter for: radiation update, Newton-Raphson iterations, max Newton-Raphson iterations, decoupled gas-dust update
amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0, 0});
int *p_iteration_failure_counter = iteration_failure_counter.data();
int *p_iteration_counter = iteration_counter.data();

// Create a MultiFab to hold radEnergySource for the current AMR level
int const nghost = 1; // depositRadiation needs 1 ghost cell
amrex::MultiFab radEnergySource(grids[lev], dmap[lev], Physics_Traits<problem_t>::nGroups, nghost);

// Implicit stage 2: solve state_tmp1 += dt * Aim_22 * g(state_tmp1)
// Full update to both radiation and gas (no gas_update_factor trick)
if constexpr (IMEX_Aim_22 > 0.0) {
    radEnergySource.setVal(0.0);
#if AMREX_SPACEDIM == 3
    particleRegister_.depositRadiation(radEnergySource, lev, time_subcycle);
#endif
    for (amrex::MFIter iter(state_tmp1_cc); iter.isValid(); ++iter) {
        const amrex::Box &indexRange = iter.validbox();
        auto const &stateTmp1 = state_tmp1_cc.array(iter);
        auto const &prob_lo = geom[lev].ProbLoArray();
        auto const &prob_hi = geom[lev].ProbHiArray();
        auto const &radEnergySource_arr = radEnergySource.array(iter);
        RadSystem<problem_t>::SetRadEnergySource(radEnergySource_arr, indexRange, dx, prob_lo, prob_hi, time_subcycle + dt_radiation);
        const double dt_stage2_implicit = IMEX_Aim_22 * dt_radiation;
        const double gas_update_factor_stage2 = 1.0; // full update with temp storage
        if constexpr (Physics_Traits<problem_t>::nGroups <= 1) {
            RadSystem<problem_t>::AddSourceTermsSingleGroup(stateTmp1, radEnergySource_arr, indexRange, dt_stage2_implicit,
                                                            gas_update_factor_stage2, dustGasInteractionCoeff_, rad_tol, rad_tol_rel,
                                                            tempFloor, p_iteration_counter, p_iteration_failure_counter);
        } else {
            RadSystem<problem_t>::AddSourceTermsMultiGroup(stateTmp1, radEnergySource_arr, indexRange, dt_stage2_implicit,
                                                           gas_update_factor_stage2, dustGasInteractionCoeff_, rad_tol, rad_tol_rel,
                                                           tempFloor, p_iteration_counter, p_iteration_failure_counter);
        }
    }
}

// === IMEX Stage 3 ===
// Explicit (Shu-Osher form for radiation):
//   state_new_rad = (1-alpha)*state_old_rad + alpha*state_tmp1_rad
//                   + dt*(Aex_31-alpha*Aex_21)*s(state_old_rad) + dt*Aex_32*s(state_tmp1_rad)
advanceRadiationMidpointRK2(lev, time_subcycle, dt_radiation, i, nsubSteps, fr_as_crse, fr_as_fine, state_tmp1_cc);

// Apply Shu-Osher combination to gas variables (NOT handled by AddFluxesRK2, which only touches radiation)
// state_new_gas = (1-alpha)*state_new_gas + alpha*state_tmp1_gas
// For variables unchanged by AddSourceTerms (e.g. density), state_new == state_tmp1, so this is a no-op.
if constexpr (nstartHyperbolic_ > 0) {
    amrex::MultiFab::LinComb(state_new_cc_[lev],
        1.0 - IMEX_alpha, state_new_cc_[lev], 0,
        IMEX_alpha, state_tmp1_cc, 0,
        0, nstartHyperbolic_, 0);
}
// Also combine any components after the radiation hyperbolic range (scalars, dust, etc.)
{
    const int post_start = nstartHyperbolic_ + ncompHyperbolic_;
    const int post_count = state_new_cc_[lev].nComp() - post_start;
    if (post_count > 0) {
        amrex::MultiFab::LinComb(state_new_cc_[lev],
            1.0 - IMEX_alpha, state_new_cc_[lev], post_start,
            IMEX_alpha, state_tmp1_cc, post_start,
            post_start, post_count, 0);
    }
}

// Implicit stage 3: solve state_new += dt * Aim_33 * g(state_new)
radEnergySource.setVal(0.0);
#if AMREX_SPACEDIM == 3
particleRegister_.depositRadiation(radEnergySource, lev, time_subcycle);
#endif
for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &stateNew_cc = state_new_cc_[lev].array(iter);
    auto const &prob_lo = geom[lev].ProbLoArray();
    auto const &prob_hi = geom[lev].ProbHiArray();
    auto const &radEnergySource_arr = radEnergySource.array(iter);
    RadSystem<problem_t>::SetRadEnergySource(radEnergySource_arr, indexRange, dx, prob_lo, prob_hi, time_subcycle + dt_radiation);
    const double dt_stage3_implicit = IMEX_Aim_33 * dt_radiation;
    const double gas_update_factor_stage3 = 1.0; // full update
    if constexpr (Physics_Traits<problem_t>::nGroups <= 1) {
        RadSystem<problem_t>::AddSourceTermsSingleGroup(stateNew_cc, radEnergySource_arr, indexRange, dt_stage3_implicit,
                                                        gas_update_factor_stage3, dustGasInteractionCoeff_, rad_tol, rad_tol_rel,
                                                        tempFloor, p_iteration_counter, p_iteration_failure_counter);
    } else {
        RadSystem<problem_t>::AddSourceTermsMultiGroup(stateNew_cc, radEnergySource_arr, indexRange, dt_stage3_implicit,
                                                       gas_update_factor_stage3, dustGasInteractionCoeff_, rad_tol, rad_tol_rel,
                                                       tempFloor, p_iteration_counter, p_iteration_failure_counter);
    }
}
```

**Note:** The `iteration_failure_counter`, `iteration_counter`, `radEnergySource` declarations were moved inside the substep loop (from before the old stage 1) since they are still needed per substep. Their handling after the source terms (lines 3045-3107) remains unchanged.

**Note on reflux scaling:**
The current `advanceRadiationForwardEuler` calls `incrementFluxRegisters(... , 0.5 * dt_radiation)` and `advanceRadiationMidpointRK2` also calls `incrementFluxRegisters(... , 0.5 * dt_radiation)`. These `0.5` factors are the quadrature weights bex_1 and bex_2. Since bex = (0.5, 0.5, 0) for PD-ARS, the values are unchanged. For future generalization, these should be parameterized.

- [ ] **Step 5.4: Update declarations in the class definition**

Update the member function declarations (lines 358-361 of QuokkaSimulation.hpp) to match the new signatures:

```cpp
void advanceRadiationForwardEuler(int lev, amrex::Real time, amrex::Real dt_radiation, int iter_count, int nsubsteps,
                                   amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine,
                                   amrex::MultiFab &state_out);
void advanceRadiationMidpointRK2(int lev, amrex::Real time, amrex::Real dt_radiation, int iter_count, int nsubsteps,
                                  amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine,
                                  amrex::MultiFab &state_inter);
```

- [ ] **Step 5.5: Verify MultiFab::LinComb signature**

Before committing, verify that `amrex::MultiFab::LinComb` supports the 10-argument form:
```cpp
static void LinComb(MultiFab &dst, Real a, MultiFab const &x, int xcomp, Real b, MultiFab const &y, int ycomp, int dstcomp, int numcomp, int nghost);
```

If this signature is not available in the AMReX version used, replace with an equivalent ParallelFor loop:
```cpp
for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &stateNew = state_new_cc_[lev].array(iter);
    auto const &stateTmp = state_tmp1_cc.const_array(iter);
    amrex::ParallelFor(indexRange, nstartHyperbolic_, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
        stateNew(i, j, k, n) = (1.0 - IMEX_alpha) * stateNew(i, j, k, n) + IMEX_alpha * stateTmp(i, j, k, n);
    });
}
```

- [ ] **Step 5.6: Commit**

```bash
git add src/QuokkaSimulation.hpp
git commit -m "refactor: rewrite radiation IMEX subcycling with Butcher tableau

Use named Butcher tableau constexpr entries and temporary state storage
(state_tmp1_cc) instead of the gas_update_factor half-step backward trick.
Stage 2 applies full implicit update to temporary. Stage 3 uses Shu-Osher
form for radiation (via AddFluxesRK2) and explicit LinComb for gas variables.

Single-group: algebraically equivalent to previous implementation.
Multi-group: gas_update_factor=1.0 replaces the 0.5 approximation in the
work-term iteration, implementing the correct IMEX formulation."
```

---

### Task 6: Build and Validate

**Files:** None (testing only)

- [ ] **Step 6.1: Build a radiation test problem**

Run: `cd $(pwd)/tests && JOB=RadMarshak make b 2>&1 | tail -30`
Expected: Clean compilation with no errors. If there are compilation errors, fix them and rebuild.

- [ ] **Step 6.2: Run the radiation test**

Run: `cd $(pwd)/tests && JOB=RadMarshak make r 2>&1 | tail -20`
Expected: Test passes. For single-group tests, results should be identical (or within floating-point tolerance due to operation reordering). The L-inf error should be within the test's tolerance.

- [ ] **Step 6.3: Build and run additional single-group radiation tests**

```bash
cd $(pwd)/tests
JOB=RadMarshakCGS make b && JOB=RadMarshakCGS make r
JOB=RadShadow make b && JOB=RadShadow make r
```

- [ ] **Step 6.4: Build and run a multi-group radiation test**

This is critical since the multi-group behavior intentionally changes (`gas_update_factor=1.0` replaces `0.5`):

```bash
cd $(pwd)/tests
JOB=RadTube make b && JOB=RadTube make r
```

If `RadTube` fails, the error tolerance may need adjustment since the multi-group implicit solve now uses the mathematically correct `gas_update_factor=1.0`. Compare error magnitudes to verify the failure is due to the expected behavioral change, not a bug.

- [ ] **Step 6.5: Build and run the default SN test**

```bash
cd $(pwd)/tests
make b && make r
```

- [ ] **Step 6.6: Commit any fixes**

If any fixes were needed, commit them:
```bash
git add -u
git commit -m "fix: address compilation/test issues from IMEX refactor"
```

---

### Task 7: Cleanup and Documentation

- [ ] **Step 7.1: Verify no remaining references to old constants**

Search for any remaining references to `IMEX_a22` or `IMEX_a32` in the source tree:
```bash
grep -rn 'IMEX_a22\|IMEX_a32' src/
```
Expected: No matches.

- [ ] **Step 7.2: Write PR.md**

Create `PR.md` in the repo root with a summary of the changes.

- [ ] **Step 7.3: Final commit**

```bash
git add -u
git commit -m "docs: add PR description for IMEX Butcher tableau refactor"
```
