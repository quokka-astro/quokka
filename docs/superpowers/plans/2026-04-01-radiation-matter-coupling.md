# Radiation-Matter Coupling Reimplementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Quokka's fragmented radiation-matter coupling source-term implementation (4 duplicated solver paths across 3 files) with a unified, modular design specified in `docs/superpowers/design.md`.

**Architecture:** Six new files in `src/radiation/` replace three old ones. The design uses compile-time dispatch (gas-only vs dust) and runtime dispatch (coupled vs decoupled) with free functions + structs. The solver is unified across single-group and multi-group via `constexpr if`. An operator-split Step A handles chemical-band attenuation before the thermal Newton solve in Step B.

**Tech Stack:** C++20, AMReX (GPU-portable `ParallelFor`), template metaprogramming for compile-time dispatch. Build with CMake/Ninja via the `quokka` CLI tool.

**Specs:**
- Physics & algorithm: `docs/superpowers/physics.md`
- Module design: `docs/superpowers/design.md`
- Test matrix: `docs/superpowers/test_problems.md`

**Build/test workflow:**
```bash
REPO_ROOT=$(pwd)
source ~/.local/bin/quokka.rc
quokka build <preset> <TestName> --root "$REPO_ROOT"
quokka run   <preset> <TestName> --root "$REPO_ROOT"
```
Presets: `1d`, `3d`. Use `1d` unless the problem is 3D-only.

**Important context:**
- All new files are header-only `.hpp` files included by `radiation_system.hpp` (existing pattern)
- All functions that run inside `ParallelFor` need `AMREX_GPU_DEVICE` annotation
- `quokka::valarray<double, N>` is a GPU-friendly fixed-size array with arithmetic operators
- The `RadSystem<problem_t>` class template owns the static member functions and compile-time constants
- Language server shows many false-positive errors for AMReX includes — ignore them, only trust build output

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/radiation/coupling_types.hpp` | **Create** | `ChemicalBandRole`, `Chemistry_Traits`, `DustModel`, `CouplingState`, `RadMatterContext`, `NewtonIterateState`, `SolverParams`, `ThermalResult`, `FluxResult`, `DiagnosticTrace` |
| `src/radiation/opacity_evaluation.hpp` | **Create** | `EvaluateOpacities`, `EvaluateFluxOpacities` — all opacity-model branching |
| `src/radiation/dust_closure.hpp` | **Create** | `SelectDustModel`, `ComputeDustTemperature`, `ComputeInitialDustTemperature` |
| `src/radiation/thermal_solve.hpp` | **Create** | `SolveRadiationMatterCoupling`, 3 Jacobian functions, `SolveArrowheadSystem`, line-search damping |
| `src/radiation/flux_update.hpp` | **Create** | `UpdateFluxAndMomentum`, `ComputeWorkTerm`, `WorkConverged` |
| `src/radiation/source_terms.hpp` | **Create** | `AddSourceTerms` — unified orchestration replacing single-group and multi-group entry points |
| `src/radiation/radiation_system.hpp` | **Modify** | Add `#include` for new files, add `Chemistry_Traits` derived constants, replace `AddSourceTermsSingleGroup`/`AddSourceTermsMultiGroup` declarations with unified `AddSourceTerms`, remove old struct declarations that move to `coupling_types.hpp` |
| `src/QuokkaSimulation.hpp` | **Modify** | Replace call sites: `AddSourceTermsSingleGroup`/`AddSourceTermsMultiGroup` → `AddSourceTerms` |
| `src/radiation/source_terms_single_group.hpp` | **Delete** | Replaced by `source_terms.hpp` + `thermal_solve.hpp` |
| `src/radiation/source_terms_multi_group.hpp` | **Delete** | Replaced by `source_terms.hpp` + `thermal_solve.hpp` + `opacity_evaluation.hpp` |
| `src/radiation/radiation_dust_system.hpp` | **Delete** | Replaced by `thermal_solve.hpp` + `dust_closure.hpp` |

---

## Test Matrix

Tests are organized by which physics path they exercise. Run these in order of complexity.

**Tier 1 — Gas-only, single-group (simplest path through new code):**
- `RadStreaming` (1D, no hydro, SG)
- `RadMarshakAsymptotic` (1D, no hydro, SG)
- `RadhydroShockCGS` (1D, hydro, SG — most stringent RHD test)
- `RadhydroUniformAdvecting` (1D, hydro, SG)
- `RadForce` (1D, hydro, SG)

**Tier 2 — Gas-only, multi-group:**
- `RadMarshakVaytet` (1D, no hydro, MG)
- `RadhydroShockMultigroup` (1D, hydro, MG)
- `RadhydroPulseMGint` (1D, hydro, MG, frequency-dependent opacity)
- `RadhydroPulseMGconst` (1D, hydro, MG, constant opacity)
- `RadTube` (1D, hydro, MG)
- `RadhydroBB` (1D, hydro, MG, PPL model accuracy)

**Tier 3 — Dust coupling:**
- `RadDust` (1D, hydro, SG+ThermalDust)
- `RadDustMG` (1D, hydro, MG+ThermalDust)
- `RadMarshakDust` (1D, no hydro, MG+ThermalDust)

**Tier 4 — Dust + PE (validates Step A + Chemistry_Traits):**
- `RadMarshakDustPE` (1D, no hydro, MG+ThermalDust+PE)

**Tier 5 — 3D + external source:**
- `ParticleRadiation` (3D, hydro, MG, stellar particles — tests `Src[n]`)

---

## Task 1: Create `coupling_types.hpp` — Data Structures and Chemistry_Traits

**Files:**
- Create: `src/radiation/coupling_types.hpp`

All data structures from the design doc. No logic beyond `constexpr` helpers. This file has no dependencies on `radiation_system.hpp` — it only needs AMReX types and `quokka::valarray`.

- [ ] **Step 1: Create `coupling_types.hpp` with `ChemicalBandRole` and `Chemistry_Traits`**

```cpp
// src/radiation/coupling_types.hpp
#ifndef COUPLING_TYPES_HPP_
#define COUPLING_TYPES_HPP_

#include "AMReX_GpuQualifiers.H"
#include "AMReX_Array.H"
#include "math/quadrature.hpp" // for quokka::valarray
#include <array>

// Forward declaration: OpacityTerms is still defined in radiation_system.hpp for now.
// It will be referenced by ThermalResult and NewtonIterateState via template parameter.

enum class ChemicalBandRole { PE, HI_ion, HeI_ion, HeII_ion };

/// Default Chemistry_Traits: no chemical bands.
/// Problems with chemical bands specialize this struct.
template <typename problem_t>
struct Chemistry_Traits {
	static constexpr int nChemicalGroups = 0;
	static constexpr std::array<ChemicalBandRole, 0> chemical_band_roles = {};
};

namespace detail
{
/// Find the global group index of a chemical band with the given role.
/// Returns -1 if not found.
template <typename problem_t>
constexpr auto FindChemicalBand(ChemicalBandRole role, int nThermalGroups) -> int
{
	constexpr auto &roles = Chemistry_Traits<problem_t>::chemical_band_roles;
	for (int i = 0; i < Chemistry_Traits<problem_t>::nChemicalGroups; ++i) {
		if (roles[i] == role) {
			return nThermalGroups + i;
		}
	}
	return -1;
}

/// Count occurrences of a chemical band role.
template <typename problem_t>
constexpr auto CountChemicalBand(ChemicalBandRole role) -> int
{
	constexpr auto &roles = Chemistry_Traits<problem_t>::chemical_band_roles;
	int count = 0;
	for (int i = 0; i < Chemistry_Traits<problem_t>::nChemicalGroups; ++i) {
		if (roles[i] == role) {
			++count;
		}
	}
	return count;
}

/// Check that all chemical band roles are unique.
template <typename problem_t>
constexpr auto AllUniqueRoles() -> bool
{
	constexpr auto &roles = Chemistry_Traits<problem_t>::chemical_band_roles;
	for (int i = 0; i < Chemistry_Traits<problem_t>::nChemicalGroups; ++i) {
		for (int j = i + 1; j < Chemistry_Traits<problem_t>::nChemicalGroups; ++j) {
			if (roles[i] == roles[j]) {
				return false;
			}
		}
	}
	return true;
}
} // namespace detail

enum class DustModel { gas_only, coupled, decoupled };

/// Solver control parameters. Not per-cell.
struct SolverParams {
	double resid_tol;
	double rel_change_tol;
	int max_newton_iter;
	int max_outer_iter;
};

/// Structured per-cell debug output. Compiled away when debug_mode = false.
template <bool enabled, int nGroups> struct DiagnosticTrace {
};

template <int nGroups> struct DiagnosticTrace<true, nGroups> {
	static constexpr int max_recorded_iters = 20;
	int n_recorded = 0;
	struct IterationSnapshot {
		double Egas;
		double T_gas;
		double T_d;
		quokka::valarray<double, nGroups> Rvec;
		quokka::valarray<double, nGroups> Erad;
		double F0;
		double Fg_abs_sum;
		double damping_factor;
	};
	amrex::GpuArray<IterationSnapshot, max_recorded_iters> snapshots;
};

// Note: CouplingState, RadMatterContext, NewtonIterateState, ThermalResult, FluxResult
// are templated on problem_t (they use nGroups_ and OpacityTerms<problem_t>).
// They are defined as nested types or free structs in the radiation module headers
// that have access to the RadSystem constants. See thermal_solve.hpp and source_terms.hpp.

#endif // COUPLING_TYPES_HPP_
```

- [ ] **Step 2: Verify it compiles**

Add `#include "radiation/coupling_types.hpp"` at the top of `src/radiation/radiation_system.hpp` (before the `RadSystem` class definition). Build any Tier 1 test:

```bash
source ~/.local/bin/quokka.rc
quokka build 1d RadStreaming --root "$REPO_ROOT"
```

Expected: compiles successfully. The new header is included but not yet used.

- [ ] **Step 3: Add derived constants to `RadSystem`**

In `src/radiation/radiation_system.hpp`, inside the `RadSystem` class, after the existing `nGroups_` constant, add:

```cpp
static constexpr int nChemicalGroups_ = Chemistry_Traits<problem_t>::nChemicalGroups;
static constexpr int nThermalGroups_ = nGroups_ - nChemicalGroups_;
static constexpr int PE_group_index_ = detail::FindChemicalBand<problem_t>(ChemicalBandRole::PE, nThermalGroups_);
static constexpr bool has_PE_heating_ = (PE_group_index_ >= 0);

static_assert(nChemicalGroups_ >= 0 && nChemicalGroups_ <= nGroups_);
static_assert(nThermalGroups_ >= 1, "Must have at least one thermal group");
static_assert(detail::CountChemicalBand<problem_t>(ChemicalBandRole::PE) <= 1, "At most one PE band allowed");
static_assert(detail::AllUniqueRoles<problem_t>(), "Chemical band roles must be unique");
```

- [ ] **Step 4: Rebuild all Tier 1 + Tier 2 tests to verify `static_assert` passes for all existing problems**

```bash
quokka build 1d RadStreaming --root "$REPO_ROOT"
quokka build 1d RadhydroShockCGS --root "$REPO_ROOT"
quokka build 1d RadMarshakVaytet --root "$REPO_ROOT"
quokka build 1d RadhydroShockMultigroup --root "$REPO_ROOT"
```

Expected: all compile. Default `Chemistry_Traits` gives `nChemicalGroups = 0`, `nThermalGroups_ = nGroups_`, `PE_group_index_ = -1`.

- [ ] **Step 5: Build a dust+PE test to verify `static_assert` passes**

```bash
quokka build 1d RadMarshakDustPE --root "$REPO_ROOT"
```

Expected: compiles. `RadMarshakDustPE` doesn't specialize `Chemistry_Traits` yet (that happens in Task 6), but the default `Chemistry_Traits` is valid.

- [ ] **Step 6: Commit**

```bash
git add src/radiation/coupling_types.hpp src/radiation/radiation_system.hpp
git commit -m "rad: add coupling_types.hpp with Chemistry_Traits and derived constants"
```

---

## Task 2: Create `opacity_evaluation.hpp` — Opacity Interface

**Files:**
- Create: `src/radiation/opacity_evaluation.hpp`
- Modify: `src/radiation/radiation_system.hpp` (add `#include`)

Extract `ComputeModelDependentKappaEAndKappaP` and `ComputeModelDependentKappaFAndDeltaTerms` from `source_terms_multi_group.hpp` into free-standing functions. The opacity-model branching (`piecewise_constant_opacity`, `PPL_opacity_fixed_slope_spectrum`, `PPL_opacity_full_spectrum`) is fully encapsulated here.

**Important:** At this stage the old code still calls the old functions. The new file provides parallel implementations that will be wired in during Task 6. We verify they produce identical results by adding temporary assertions in Task 4.

- [ ] **Step 1: Create `opacity_evaluation.hpp`**

Extract the two functions from `source_terms_multi_group.hpp` (lines 8-95) into `src/radiation/opacity_evaluation.hpp`. Keep them as `RadSystem<problem_t>` static member functions for now (changing to free functions is a later cleanup). The key change: wrap them in a cleaner interface that takes `NewtonIterateState`-compatible arguments.

```cpp
// src/radiation/opacity_evaluation.hpp
// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef OPACITY_EVALUATION_HPP_
#define OPACITY_EVALUATION_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// EvaluateOpacities: wraps ComputeModelDependentKappaEAndKappaP
// This is the opacity interface for the new thermal solver.
// All opacity-model branching is internal.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::EvaluateOpacities(
    double T_d, double rho,
    quokka::valarray<double, nGroups_> const &Erad,
    int iteration_number,
    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
    amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
    quokka::valarray<double, nGroups_> const &fourPiBoverC,
    OpacityTerms<problem_t> const &prev_opacity) -> OpacityTerms<problem_t>
{
    return ComputeModelDependentKappaEAndKappaP(T_d, rho, rad_boundaries, rad_boundary_ratios,
                                                 fourPiBoverC, Erad, iteration_number,
                                                 prev_opacity.alpha_E, prev_opacity.alpha_P);
}

// EvaluateFluxOpacities: wraps ComputeModelDependentKappaFAndDeltaTerms
template <typename problem_t>
AMREX_GPU_DEVICE void RadSystem<problem_t>::EvaluateFluxOpacities(
    double T_d, double rho,
    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
    quokka::valarray<double, nGroups_> const &fourPiBoverC,
    OpacityTerms<problem_t> &opacity_terms)
{
    ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms);
}

#endif // OPACITY_EVALUATION_HPP_
```

- [ ] **Step 2: Declare the new functions in `RadSystem` class**

In `src/radiation/radiation_system.hpp`, inside the `RadSystem` class declaration, add:

```cpp
AMREX_GPU_DEVICE static auto EvaluateOpacities(double T_d, double rho,
    quokka::valarray<double, nGroups_> const &Erad, int iteration_number,
    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
    amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
    quokka::valarray<double, nGroups_> const &fourPiBoverC,
    OpacityTerms<problem_t> const &prev_opacity) -> OpacityTerms<problem_t>;

AMREX_GPU_DEVICE static void EvaluateFluxOpacities(double T_d, double rho,
    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
    quokka::valarray<double, nGroups_> const &fourPiBoverC,
    OpacityTerms<problem_t> &opacity_terms);
```

Add `#include "radiation/opacity_evaluation.hpp"` at the bottom of `radiation_system.hpp` (after the class definition, alongside the existing `#include` for `source_terms_*.hpp`).

- [ ] **Step 3: Verify it compiles**

```bash
quokka build 1d RadhydroShockCGS --root "$REPO_ROOT"
quokka build 1d RadhydroPulseMGint --root "$REPO_ROOT"
quokka build 1d RadhydroBB --root "$REPO_ROOT"
```

Expected: compiles. The new functions are thin wrappers — no behavioral change.

- [ ] **Step 4: Commit**

```bash
git add src/radiation/opacity_evaluation.hpp src/radiation/radiation_system.hpp
git commit -m "rad: add opacity_evaluation.hpp wrapping opacity interface"
```

---

## Task 3: Create `dust_closure.hpp` — Dust Temperature and Model Selection

**Files:**
- Create: `src/radiation/dust_closure.hpp`
- Modify: `src/radiation/radiation_system.hpp` (add `#include`, add declarations)

Extract `ComputeDustTemperatureBateKeto` from `radiation_dust_system.hpp` and `source_terms_single_group.hpp`. Add `SelectDustModel` (compile-time gas-only vs runtime coupled/decoupled) and `ComputeDustTemperatureFromIterate` (dust path only — gas-only is handled by the caller setting Td = T_gas). The dust temperature computation sums radiation energy from ALL groups (thermal + chemical) because chemical-band photons absorbed by dust heat it.

- [ ] **Step 1: Create `dust_closure.hpp`**

```cpp
// src/radiation/dust_closure.hpp
// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef DUST_CLOSURE_HPP_
#define DUST_CLOSURE_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

/// Select dust model based on coupling strength.
/// Returns gas_only when enable_dust_gas_thermal_coupling_model_ is false (constexpr path).
/// Otherwise returns coupled or decoupled based on threshold.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SelectDustModel(
    double T_gas0, double T_d0, double Egas0, double coeff_n) -> DustModel
{
    if constexpr (!enable_dust_gas_thermal_coupling_model_) {
        return DustModel::gas_only;
    } else {
        const double cscale = c_light_ / c_hat_;
        const double max_Gamma_gd = coeff_n * std::max(std::sqrt(T_gas0) * T_gas0, std::sqrt(T_d0) * T_d0);
        if (cscale * max_Gamma_gd < ISM_Traits<problem_t>::gas_dust_coupling_threshold * Egas0) {
            return DustModel::decoupled;
        }
        return DustModel::coupled;
    }
}

/// Compute dust temperature from the current Newton iterate.
/// When enable_dust_gas_thermal_coupling_model_ is false, SelectDustModel returns gas_only
/// and the caller sets Td = T_gas directly — this function is only called for the dust path.
///
/// - coupled: Td = T_gas - sum(R_all) / (Nd * sqrt(T_gas))
///   where R_all sums over ALL groups (thermal + chemical), because chemical-band
///   photons absorbed by dust heat the dust.
/// - decoupled: at n==0, use T_d0; thereafter, Td is updated by the Newton step (delta_x)
///
/// Enforces tempFloor_ as dust temperature floor.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeDustTemperatureFromIterate(
    DustModel model, double T_gas,
    quokka::valarray<double, nGroups_> const &Rvec_all,
    double coeff_n, double T_d0, int newton_iter) -> double
{
    // Note: when enable_dust_gas_thermal_coupling_model_ is true, model is always
    // coupled or decoupled (never gas_only), so no need to check for gas_only here.
    double T_d = NAN;

    if (model == DustModel::coupled) {
        if (newton_iter == 0) {
            T_d = T_d0;
        } else {
            // sum over ALL groups (thermal + chemical)
            T_d = T_gas - sum(Rvec_all) / (coeff_n * std::sqrt(T_gas));
        }
    } else { // decoupled
        if (newton_iter == 0) {
            T_d = T_d0;
        }
        // For decoupled model at newton_iter > 0, T_d is updated by the Newton
        // step (delta_x is applied to T_d directly). This is handled in the caller.
    }

    // Enforce dust temperature floor
    if (T_d < tempFloor_) {
        T_d = tempFloor_;
    }

    return T_d;
}

#endif // DUST_CLOSURE_HPP_
```

- [ ] **Step 2: Declare the new functions in `RadSystem` class**

In `src/radiation/radiation_system.hpp`, inside the `RadSystem` class declaration, add:

```cpp
AMREX_GPU_DEVICE static auto SelectDustModel(double T_gas0, double T_d0, double Egas0, double coeff_n) -> DustModel;

AMREX_GPU_DEVICE static auto ComputeDustTemperatureFromIterate(
    DustModel model, double T_gas,
    quokka::valarray<double, nGroups_> const &Rvec,
    double coeff_n, double T_d0, int newton_iter) -> double;
```

Add `#include "radiation/dust_closure.hpp"` alongside the other new includes.

- [ ] **Step 3: Build dust-path tests**

```bash
quokka build 1d RadDust --root "$REPO_ROOT"
quokka build 1d RadDustMG --root "$REPO_ROOT"
quokka build 1d RadMarshakDust --root "$REPO_ROOT"
```

Expected: compiles. New functions are defined but not called yet.

- [ ] **Step 4: Commit**

```bash
git add src/radiation/dust_closure.hpp src/radiation/radiation_system.hpp
git commit -m "rad: add dust_closure.hpp with SelectDustModel and ComputeDustTemperatureFromIterate"
```

---

## Task 4: Create `thermal_solve.hpp` — Newton Solver

**Files:**
- Create: `src/radiation/thermal_solve.hpp`
- Modify: `src/radiation/radiation_system.hpp` (add `#include`, add declarations)

This is the core of the reimplementation. It contains:
1. `SolveRadiationMatterCoupling` — the Newton loop
2. Three Jacobian functions: `ComputeJacobianGasOnly`, `ComputeJacobianDustCoupled`, `ComputeJacobianDustDecoupled`
3. `SolveArrowheadSystem` — the arrowhead linear solver (port of existing `SolveLinearEqs`)
4. Line-search damping (replaces `enable_dE_constrain`)
5. `DiagnosticTrace` recording

The implementation ports logic from:
- `SolveGasRadiationEnergyExchange` in `source_terms_multi_group.hpp`
- `SolveGasDustRadiationEnergyExchange` in `radiation_dust_system.hpp`
- The single-group Newton loop in `source_terms_single_group.hpp`
- `ComputeJacobianForGas`, `ComputeJacobianForGasAndDust`, `ComputeJacobianForGasAndDustDecoupled`

The key changes from old code:
- Store `R[n]` directly (no `D[n]` / `use_D_as_base`)
- Unified single-group/multi-group via `constexpr if (nGroups_ == 1)`
- Three Jacobian functions take `(NewtonIterateState, RadMatterContext)` instead of 12+ individual parameters
- Line-search damping replaces `enable_dE_constrain`
- Newton loop updates only the `nThermalGroups_` radiation unknowns (not `nGroups_`)
- **Dust temperature uses ALL bands:** The dust temperature computation (`ComputeDustTemperatureFromIterate`) and opacity evaluation use total radiation energy from all `nGroups_` bands (thermal + advanced chemical), because chemical-band photons absorbed by dust heat it. The thermal Jacobian and residual vectors are `nThermalGroups_`-sized, but the dust temperature closure sums over all groups.
- **Work term includes chemical bands:** After Step A advances chemical bands, their contribution enters the work-lag outer loop via the full `nGroups_`-sized flux array. The work term `w[g]` is computed for all groups.

**This task is large. Implement it in sub-steps, building and testing incrementally.**

- [ ] **Step 1: Define `CouplingState`, `RadMatterContext`, `NewtonIterateState`, `ThermalResult`, `FluxResult` structs**

Add these to `src/radiation/coupling_types.hpp` as templates parameterized on `problem_t` (they depend on `nGroups_` and `OpacityTerms<problem_t>`). Since they need `RadSystem` constants, define them inside `radiation_system.hpp` after the `RadSystem` class, or as nested types. The cleanest approach: define them in `coupling_types.hpp` with explicit template parameters for `nGroups` and `nmscalars`:

Add to `coupling_types.hpp`:

```cpp
template <int nGroups, int nmscalars>
struct CouplingState {
	double rho;
	double Egas0;
	double Ekin0;
	amrex::GpuArray<double, 3> gasMomentum0;
	quokka::valarray<double, nGroups> Erad0;
	quokka::valarray<double, nGroups> Src;
	amrex::GpuArray<double, 3 * nGroups> Frad0_flat; // flattened [dim * nGroups]
	amrex::GpuArray<amrex::Real, nmscalars> massScalars;
	double dt;
	double Etot0;
};

template <int nGroups>
struct RadMatterContext_t {
	double rho;
	double Egas0;
	double Ekin0;
	amrex::GpuArray<double, 3> gasMomentum0;
	quokka::valarray<double, nGroups> Erad0;
	quokka::valarray<double, nGroups> Src;
	amrex::GpuArray<double, 3 * nGroups> Frad0_flat;
	double dt;
	double Etot0;
	DustModel dust_model;
	double coeff_n;
	double lambda_gd_times_dt;
	double T_d0;
};
```

Alternatively, since these structs are only used inside `RadSystem<problem_t>` member functions that already have access to `nGroups_`, define them as type aliases inside `RadSystem`:

```cpp
// Inside RadSystem<problem_t> class:
using CouplingState_t = CouplingState<nGroups_, nmscalars_>;
using RadMatterContext_t = RadMatterContext_t<nGroups_>;
```

Choose whichever approach compiles cleanly. The exact layout can be adjusted — what matters is that the structs are GPU-copyable (no pointers, no virtual functions).

- [ ] **Step 2: Create `thermal_solve.hpp` with `SolveArrowheadSystem`**

Port the existing `SolveLinearEqs` from `radiation_system.hpp` (lines 585-590). This is a direct copy with a renamed function:

```cpp
// src/radiation/thermal_solve.hpp
// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef THERMAL_SOLVE_HPP_
#define THERMAL_SOLVE_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

/// Solve the arrowhead linear system:
///   [J00 J0g] [x0]   [F0]
///   [Jg0 Jgg] [xg] = [Fg]
/// where J0g, Jg0, Jgg are vectors (diagonal + first row + first column structure).
template <typename problem_t>
AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveArrowheadSystem(
    JacobianResult<problem_t> const &jacobian, double &x0,
    quokka::valarray<double, nGroups_> &xi)
{
    // Same algorithm as existing SolveLinearEqs
    const auto ratios = jacobian.J0g / jacobian.Jgg;
    x0 = (sum(ratios * jacobian.Fg) - jacobian.F0) / (-sum(ratios * jacobian.Jg0) + jacobian.J00);
    xi = (-1.0 * jacobian.Fg - jacobian.Jg0 * x0) / jacobian.Jgg;
}

#endif // THERMAL_SOLVE_HPP_
```

- [ ] **Step 3: Add `ComputeJacobianGasOnly`**

Port from `ComputeJacobianForGas` in `source_terms_multi_group.hpp` (lines 105-147). The key simplification: `dTd/dT = 1`, `dTd/dR = 0`, no dust terms. Uses `nThermalGroups_` instead of `nGroups_` for the residual computation.

```cpp
/// Gas-only Jacobian: Td = T_gas, no dust coupling terms.
/// Temperature derivatives of kappaP/kappaE are neglected (affects convergence rate only).
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianGasOnly(
    double Egas_diff,
    quokka::valarray<double, nThermalGroups_> const &Erad_diff,
    quokka::valarray<double, nThermalGroups_> const &Rvec,
    quokka::valarray<double, nThermalGroups_> const &Src,
    quokka::valarray<double, nThermalGroups_> const &tau,
    double c_v,
    quokka::valarray<double, nThermalGroups_> const &kappaPoverE,
    quokka::valarray<double, nThermalGroups_> const &d_fourpiboverc_d_t,
    double num_den, double dt) -> JacobianResult<problem_t>
{
    JacobianResult<problem_t> result;
    const double cscale = c_light_ / c_hat_;
    const double CR_heating = DefineCosmicRayHeatingRate(num_den) * dt;

    result.F0 = Egas_diff + cscale * sum(Rvec) - CR_heating;
    result.Fg = Erad_diff - (Rvec + Src);
    result.Fg_abs_sum = 0.0;
    for (int g = 0; g < nThermalGroups_; ++g) {
        if (tau[g] > 0.0) {
            result.Fg_abs_sum += std::abs(result.Fg[g]);
        }
    }

    // Jacobian elements. d/dT(kappaP/kappaE) is neglected.
    const auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t;
    result.J00 = 1.0;
    result.J0g.fillin(cscale);
    result.Jg0 = 1.0 / c_v * dEg_dT;
    for (int g = 0; g < nThermalGroups_; ++g) {
        if (tau[g] <= 0.0) {
            result.Jgg[g] = -std::numeric_limits<double>::infinity();
        } else {
            result.Jgg[g] = -1.0 * kappaPoverE[g] / tau[g] - 1.0;
        }
    }
    return result;
}
```

- [ ] **Step 4: Add `ComputeJacobianDustCoupled` and `ComputeJacobianDustDecoupled`**

Port from `ComputeJacobianForGasAndDust` and `ComputeJacobianForGasAndDustDecoupled` in `radiation_dust_system.hpp`. The coupled variant includes the Schur complement modification to `Fg`. Both are guarded by `if constexpr (enable_dust_gas_thermal_coupling_model_)` to avoid compilation when dust is off.

These follow the same pattern as Step 3 — port the existing logic, adapt to use `nThermalGroups_` and the design doc's Jacobian formulas.

- [ ] **Step 5: Add `SolveRadiationMatterCoupling` — the Newton loop**

This is the main function. Port from `SolveGasRadiationEnergyExchange` (gas-only path) and `SolveGasDustRadiationEnergyExchange` (dust path), unified into a single function. Key changes from old code:

1. Uses `R[n]` directly (no `D[n]` scaling)
2. Dispatches to the three Jacobian functions based on `dust_model`
3. Newton unknowns are `nThermalGroups_`-sized (Egas + R[0..nThermal-1])
4. Dust temperature uses ALL `nGroups_` bands: `Rvec_all` includes the (fixed, post-Step-A) chemical-band contributions alongside the iterated thermal-band `Rvec`. This is assembled by concatenating the thermal `Rvec` with the known chemical-band `R_chem` before calling `ComputeDustTemperatureFromIterate`.
5. Uses line-search damping instead of `enable_dE_constrain`:

```cpp
// Line-search damping (replaces enable_dE_constrain)
double alpha = 1.0;
constexpr double alpha_min = 1.0e-4;
constexpr double overshoot_factor = 4.0;
const double T_rad = std::sqrt(std::sqrt(sum(EradVec_guess) / radiation_constant_));
const double T_bound = std::max(iterate.T_gas, T_rad) * overshoot_factor;

while (alpha > alpha_min) {
    const double Egas_trial = iterate.Egas + alpha * delta_x;
    if (Egas_trial > 0.0) {
        const double T_trial = quokka::EOS<problem_t>::ComputeTgasFromEint(
            ctx.rho, Egas_trial, ctx.massScalars);
        if (T_trial > 0.0 && T_trial < T_bound) {
            break;
        }
    }
    alpha *= 0.5;
}
iterate.Egas += alpha * delta_x;
iterate.Rvec += alpha * delta_R;
```

6. Records `DiagnosticTrace` snapshots when `debug_mode = true`
7. For the decoupled dust model: post-convergence gas energy update via implicit cooling balance (port the `rhs` lambda and root-finding from `radiation_dust_system.hpp` lines 544-549)

- [ ] **Step 6: Declare all new functions in `RadSystem` class and add `#include`**

In `src/radiation/radiation_system.hpp`:
- Declare `SolveArrowheadSystem`, `ComputeJacobianGasOnly`, `ComputeJacobianDustCoupled`, `ComputeJacobianDustDecoupled`, `SolveRadiationMatterCoupling`
- Add `#include "radiation/thermal_solve.hpp"`

- [ ] **Step 7: Build Tier 1 tests (gas-only, single-group)**

```bash
quokka build 1d RadStreaming --root "$REPO_ROOT"
quokka build 1d RadhydroShockCGS --root "$REPO_ROOT"
```

Expected: compiles. New functions are defined but not called yet.

- [ ] **Step 8: Build Tier 3 tests (dust coupling)**

```bash
quokka build 1d RadDust --root "$REPO_ROOT"
quokka build 1d RadMarshakDust --root "$REPO_ROOT"
```

Expected: compiles.

- [ ] **Step 9: Commit**

```bash
git add src/radiation/thermal_solve.hpp src/radiation/coupling_types.hpp src/radiation/radiation_system.hpp
git commit -m "rad: add thermal_solve.hpp with unified Newton solver and Jacobian functions"
```

---

## Task 5: Create `flux_update.hpp` — Flux/Momentum Update and Work Term

**Files:**
- Create: `src/radiation/flux_update.hpp`
- Modify: `src/radiation/radiation_system.hpp` (add `#include`, add declarations)

Extract `UpdateFlux` from `source_terms_multi_group.hpp` and the flux update section from `source_terms_single_group.hpp`. The work-term computation and convergence check are also here.

- [ ] **Step 1: Create `flux_update.hpp`**

Port `UpdateFlux` from `source_terms_multi_group.hpp` (lines 427-586). Key changes:
- Takes `RadMatterContext` and `ThermalResult` instead of raw arrays + indices
- Operates on all `nGroups_` (thermal + chemical) for flux relaxation
- `beta_order_ >= 2` code is dropped (design decision: `<= 1` only)
- Work term computation extracted into `ComputeWorkTerm`
- Work convergence check extracted into `WorkConverged`

```cpp
// src/radiation/flux_update.hpp
// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef FLUX_UPDATE_HPP_
#define FLUX_UPDATE_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// UpdateFluxAndMomentum: update Frad and gas momentum for all groups.
// Port of UpdateFlux from source_terms_multi_group.hpp.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::UpdateFluxAndMomentum(
    /* ctx, thermal, consPrev, i, j, k, gas_update_factor, Ekin0 */)
    -> FluxUpdateResult<problem_t>
{
    // ... port existing UpdateFlux logic ...
    // Key: iterate over all nGroups_ (not just nThermalGroups_) for flux relaxation
    // For beta_order_ == 0: Frad_new = Frad_old / (1 + rho * kappaF * chat * dt)
    // For beta_order_ == 1: includes Planck and pressure terms
    // Work term correction: remove kinetic energy change from Egas when include_work_term_in_source
}

// ComputeWorkTerm: compute work[g] = (v . F[g]) * kappaF[g] * chat / c^2 * dt
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeWorkTerm(
    /* updated momentum, updated Frad, opacity_terms, dt */)
    -> quokka::valarray<double, nGroups_>
{
    quokka::valarray<double, nGroups_> work{};
    // ... port work term computation from source_terms_multi_group.hpp lines 530-541 ...
    return work;
}

// WorkConverged: check if work term has converged between outer iterations.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::WorkConverged(
    quokka::valarray<double, nGroups_> const &work,
    quokka::valarray<double, nGroups_> const &work_prev,
    double Etot0) -> bool
{
    // Port convergence check from source_terms_multi_group.hpp lines 777-788
    const double rel_lag_tol = 1.0e-8;
    const double lag_tol = 1.0e-13;
    double ref_work = rel_lag_tol * sum(abs(work));
    ref_work = std::max(ref_work, lag_tol * Etot0 / (c_light_ / c_hat_));
    return sum(abs(work - work_prev)) <= ref_work;
}

#endif // FLUX_UPDATE_HPP_
```

- [ ] **Step 2: Declare functions in `RadSystem` class and add `#include`**

- [ ] **Step 3: Build tests**

```bash
quokka build 1d RadhydroShockCGS --root "$REPO_ROOT"
quokka build 1d RadhydroShockMultigroup --root "$REPO_ROOT"
quokka build 1d RadForce --root "$REPO_ROOT"
```

Expected: compiles.

- [ ] **Step 4: Commit**

```bash
git add src/radiation/flux_update.hpp src/radiation/radiation_system.hpp
git commit -m "rad: add flux_update.hpp with UpdateFluxAndMomentum and WorkConverged"
```

---

## Task 6: Create `source_terms.hpp` — Unified Orchestration

**Files:**
- Create: `src/radiation/source_terms.hpp`
- Modify: `src/radiation/radiation_system.hpp` (replace old `#include`s and declarations)
- Modify: `src/QuokkaSimulation.hpp` (replace call sites)

This is the integration task. Wire Step A (chemical-band attenuation + PE heating) and Step B (thermal coupling via `SolveRadiationMatterCoupling` + `UpdateFluxAndMomentum` + work-lag loop) into a single `AddSourceTerms` function.

- [ ] **Step 1: Create `source_terms.hpp` with `AddSourceTerms`**

This replaces both `AddSourceTermsSingleGroup` and `AddSourceTermsMultiGroup`. The function signature matches the existing ones (same parameters) for easy call-site migration.

```cpp
// src/radiation/source_terms.hpp
// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef SOURCE_TERMS_HPP_
#define SOURCE_TERMS_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource,
                                           amrex::Box const &indexRange, Real dt_implicit,
                                           double gas_update_factor, double dustGasCoeff,
                                           double tol_h, double tol_rel_h, double tempFloor_local,
                                           int *p_iteration_counter, int *p_iteration_failure_counter)
{
    arrayconst_t &consPrev = consVar;
    array_t &consNew = consVar;
    auto dt = dt_implicit;

    amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

    SolverParams params;
    params.resid_tol = tol_h;
    params.rel_change_tol = tol_rel_h;
    params.max_newton_iter = 100;
    params.max_outer_iter = 5;

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        // --- Step A: Chemical-band update ---
        if constexpr (nChemicalGroups_ > 0) {
            // Attenuate chemical bands by dust opacity
            // ... (as specified in design.md Step A detail) ...
        }

        // --- Step B: Thermal radiation-matter-dust coupling ---

        // Load cell state (post-Step-A)
        // ... assemble CouplingState from consVar ...

        // Isothermal early return
        if constexpr (gamma_ == 1.0) {
            // flux relaxation only, no thermal solve
            // ... write back Frad and momentum ...
            return;
        }

        // Assemble RadMatterContext (immutable for this cell)
        // ... dust model selection, coeff_n computation ...

        // Outer work-lag iteration
        // Work term spans ALL nGroups_ (thermal + chemical) since the flux update
        // covers all groups and advanced chemical-band fluxes contribute to the work term.
        quokka::valarray<double, nGroups_> work{};
        quokka::valarray<double, nGroups_> work_prev{};

        for (int outer = 0; outer < params.max_outer_iter; ++outer) {
            // SolveRadiationMatterCoupling operates on nThermalGroups_ unknowns,
            // but receives advanced chemical-band Erad for dust temperature computation.
            auto extra_src = /* Src + work (thermal groups only) */;
            auto thermal = SolveRadiationMatterCoupling(/* ctx, extra_src, params */);
            // UpdateFluxAndMomentum operates on ALL nGroups_ for flux relaxation.
            auto flux = UpdateFluxAndMomentum(/* ... */);

            if constexpr (beta_order_ == 0 || !include_work_term_in_source) {
                // Write back and break
                break;
            }

            work_prev = work;
            // Work term computed for ALL nGroups_ (thermal + chemical bands).
            work = ComputeWorkTerm(/* ... */);
            if (WorkConverged(work, work_prev, /* Etot0 */)) {
                // Write back and break
                break;
            }
        }

        // Write back: Erad, Frad, gas momentum, gas energy
        // Apply gas_update_factor for IMEX staging
        // ... (port from existing write-back code) ...
    });
}

#endif // SOURCE_TERMS_HPP_
```

- [ ] **Step 2: Declare `AddSourceTerms` in `RadSystem` class**

In `src/radiation/radiation_system.hpp`, replace:
```cpp
static void AddSourceTermsMultiGroup(...);
static void AddSourceTermsSingleGroup(...);
```
with:
```cpp
static void AddSourceTerms(...);  // same parameter list
```

Replace the old `#include`s:
```cpp
// Remove these:
// #include "radiation/source_terms_single_group.hpp"
// #include "radiation/source_terms_multi_group.hpp"
// #include "radiation/radiation_dust_system.hpp"

// Add:
#include "radiation/source_terms.hpp"
```

- [ ] **Step 3: Update call sites in `QuokkaSimulation.hpp`**

Replace the two call sites (around lines 3005-3013 and 3081-3089):

```cpp
// Old:
if constexpr (Physics_Traits<problem_t>::nGroups <= 1) {
    RadSystem<problem_t>::AddSourceTermsSingleGroup(...);
} else {
    RadSystem<problem_t>::AddSourceTermsMultiGroup(...);
}

// New:
RadSystem<problem_t>::AddSourceTerms(...);
```

There are two call sites — one for stage 2 and one for stage 3 of the IMEX scheme.

- [ ] **Step 4: Build and run Tier 1 (gas-only SG) — most basic path**

```bash
quokka build 1d RadStreaming --root "$REPO_ROOT"
quokka run 1d RadStreaming --root "$REPO_ROOT"
```

Expected: test passes with identical results.

```bash
quokka build 1d RadhydroShockCGS --root "$REPO_ROOT"
quokka run 1d RadhydroShockCGS --root "$REPO_ROOT"
```

Expected: test passes. This is the most stringent single-group RHD test.

- [ ] **Step 5: Run remaining Tier 1 tests**

```bash
quokka build 1d RadMarshakAsymptotic --root "$REPO_ROOT" && quokka run 1d RadMarshakAsymptotic --root "$REPO_ROOT"
quokka build 1d RadhydroUniformAdvecting --root "$REPO_ROOT" && quokka run 1d RadhydroUniformAdvecting --root "$REPO_ROOT"
quokka build 1d RadForce --root "$REPO_ROOT" && quokka run 1d RadForce --root "$REPO_ROOT"
```

- [ ] **Step 6: Run Tier 2 tests (gas-only MG)**

```bash
quokka build 1d RadMarshakVaytet --root "$REPO_ROOT" && quokka run 1d RadMarshakVaytet --root "$REPO_ROOT"
quokka build 1d RadhydroShockMultigroup --root "$REPO_ROOT" && quokka run 1d RadhydroShockMultigroup --root "$REPO_ROOT"
quokka build 1d RadhydroPulseMGint --root "$REPO_ROOT" && quokka run 1d RadhydroPulseMGint --root "$REPO_ROOT"
quokka build 1d RadhydroPulseMGconst --root "$REPO_ROOT" && quokka run 1d RadhydroPulseMGconst --root "$REPO_ROOT"
quokka build 1d RadTube --root "$REPO_ROOT" && quokka run 1d RadTube --root "$REPO_ROOT"
quokka build 1d RadhydroBB --root "$REPO_ROOT" && quokka run 1d RadhydroBB --root "$REPO_ROOT"
```

- [ ] **Step 7: Run Tier 3 tests (dust coupling)**

```bash
quokka build 1d RadDust --root "$REPO_ROOT" && quokka run 1d RadDust --root "$REPO_ROOT"
quokka build 1d RadDustMG --root "$REPO_ROOT" && quokka run 1d RadDustMG --root "$REPO_ROOT"
quokka build 1d RadMarshakDust --root "$REPO_ROOT" && quokka run 1d RadMarshakDust --root "$REPO_ROOT"
```

- [ ] **Step 8: Add `Chemistry_Traits` specialization for PE problems and run Tier 4**

In `src/problems/RadMarshakDustPE/testRadMarshakDustPE.cpp`, add:

```cpp
template <>
struct Chemistry_Traits<MarshakProblem> {
    static constexpr int nChemicalGroups = 1;
    static constexpr std::array<ChemicalBandRole, 1> chemical_band_roles = { ChemicalBandRole::PE };
};
```

Similarly for `RadLineCoolingMG` if it uses PE heating.

```bash
quokka build 1d RadMarshakDustPE --root "$REPO_ROOT" && quokka run 1d RadMarshakDustPE --root "$REPO_ROOT"
```

Note: This test may initially fail or need adjustment because the PE heating pathway has changed from being inside the Newton solve to being operator-split (Step A). The converged result may differ slightly. If it does, verify that the new result is physically reasonable and update the test tolerance if needed.

- [ ] **Step 9: Run Tier 5 (3D + external source)**

```bash
quokka build 3d ParticleRadiation --root "$REPO_ROOT" && quokka run 3d ParticleRadiation --root "$REPO_ROOT"
```

This tests the `Src[n]` pathway and 3D compilation.

- [ ] **Step 10: Commit**

```bash
git add src/radiation/source_terms.hpp src/radiation/radiation_system.hpp src/QuokkaSimulation.hpp
git add src/problems/RadMarshakDustPE/testRadMarshakDustPE.cpp
git commit -m "rad: add unified source_terms.hpp, wire into QuokkaSimulation"
```

---

## Task 7: Delete Old Files and Clean Up

**Files:**
- Delete: `src/radiation/source_terms_single_group.hpp`
- Delete: `src/radiation/source_terms_multi_group.hpp`
- Delete: `src/radiation/radiation_dust_system.hpp`
- Modify: `src/radiation/radiation_system.hpp` (remove old `#include`s, old function declarations, old struct definitions that moved to `coupling_types.hpp`)

- [ ] **Step 1: Remove old `#include` directives from `radiation_system.hpp`**

Remove:
```cpp
#include "radiation/source_terms_single_group.hpp"
#include "radiation/source_terms_multi_group.hpp"
#include "radiation/radiation_dust_system.hpp"
```

These should already be removed if Task 6 was done correctly. Verify.

- [ ] **Step 2: Remove old function declarations from `RadSystem` class**

Remove declarations for:
- `AddSourceTermsSingleGroup`
- `AddSourceTermsMultiGroup`
- `SolveGasRadiationEnergyExchange`
- `SolveGasDustRadiationEnergyExchange`
- `SolveGasDustRadiationEnergyExchangeWithPE`
- `ComputeJacobianForGas`
- `ComputeJacobianForGasAndDust`
- `ComputeJacobianForGasAndDustDecoupled`
- `ComputeJacobianForGasAndDustWithPE`
- `SolveLinearEqs` (replaced by `SolveArrowheadSystem`)
- `SolveLinearEqsWithLastColumn` (no longer needed — PE is operator-split)
- `UpdateFlux` (replaced by `UpdateFluxAndMomentum`)
- `ComputeModelDependentKappaEAndKappaP` (internalized by `EvaluateOpacities`)
- `ComputeModelDependentKappaFAndDeltaTerms` (internalized by `EvaluateFluxOpacities`)

Also remove the `SolveLinearEqs` inline implementation (lines 580-590 of current `radiation_system.hpp`).

- [ ] **Step 3: Move `OpacityTerms`, `NewtonIterationResult`, `JacobianResult`, `FluxUpdateResult` to `coupling_types.hpp`**

If these structs are still defined in `radiation_system.hpp`, move them to `coupling_types.hpp`. Update any remaining references. If `ThermalResult` has replaced `NewtonIterationResult` and `FluxResult` has replaced `FluxUpdateResult`, remove the old structs entirely.

- [ ] **Step 4: Delete old files**

```bash
git rm src/radiation/source_terms_single_group.hpp
git rm src/radiation/source_terms_multi_group.hpp
git rm src/radiation/radiation_dust_system.hpp
```

- [ ] **Step 5: Full rebuild and test run**

Build and run ALL test tiers to verify nothing broke:

```bash
# Tier 1
quokka build 1d RadStreaming --root "$REPO_ROOT" && quokka run 1d RadStreaming --root "$REPO_ROOT"
quokka build 1d RadhydroShockCGS --root "$REPO_ROOT" && quokka run 1d RadhydroShockCGS --root "$REPO_ROOT"
quokka build 1d RadMarshakAsymptotic --root "$REPO_ROOT" && quokka run 1d RadMarshakAsymptotic --root "$REPO_ROOT"
quokka build 1d RadhydroUniformAdvecting --root "$REPO_ROOT" && quokka run 1d RadhydroUniformAdvecting --root "$REPO_ROOT"
quokka build 1d RadForce --root "$REPO_ROOT" && quokka run 1d RadForce --root "$REPO_ROOT"

# Tier 2
quokka build 1d RadMarshakVaytet --root "$REPO_ROOT" && quokka run 1d RadMarshakVaytet --root "$REPO_ROOT"
quokka build 1d RadhydroShockMultigroup --root "$REPO_ROOT" && quokka run 1d RadhydroShockMultigroup --root "$REPO_ROOT"
quokka build 1d RadhydroPulseMGint --root "$REPO_ROOT" && quokka run 1d RadhydroPulseMGint --root "$REPO_ROOT"
quokka build 1d RadhydroPulseMGconst --root "$REPO_ROOT" && quokka run 1d RadhydroPulseMGconst --root "$REPO_ROOT"
quokka build 1d RadTube --root "$REPO_ROOT" && quokka run 1d RadTube --root "$REPO_ROOT"
quokka build 1d RadhydroBB --root "$REPO_ROOT" && quokka run 1d RadhydroBB --root "$REPO_ROOT"

# Tier 3
quokka build 1d RadDust --root "$REPO_ROOT" && quokka run 1d RadDust --root "$REPO_ROOT"
quokka build 1d RadDustMG --root "$REPO_ROOT" && quokka run 1d RadDustMG --root "$REPO_ROOT"
quokka build 1d RadMarshakDust --root "$REPO_ROOT" && quokka run 1d RadMarshakDust --root "$REPO_ROOT"

# Tier 4
quokka build 1d RadMarshakDustPE --root "$REPO_ROOT" && quokka run 1d RadMarshakDustPE --root "$REPO_ROOT"

# Tier 5
quokka build 3d ParticleRadiation --root "$REPO_ROOT" && quokka run 3d ParticleRadiation --root "$REPO_ROOT"
```

- [ ] **Step 6: Commit**

```bash
git add -A src/radiation/
git commit -m "rad: remove old source-term files, complete reimplementation"
```

---

## Task 8: Write PR Documentation

**Files:**
- Create: `PR.md`

- [ ] **Step 1: Write PR.md**

```markdown
## Radiation-matter coupling reimplementation

Replaces the fragmented radiation-matter coupling source-term implementation (4 duplicated solver paths across 3 files) with a unified, modular design.

### Changes
- **Unified solver**: Single-group and multi-group paths merged via `constexpr if`
- **Modular structure**: 6 new files with clear separation of concerns (opacity evaluation, dust closure, thermal solve, flux update, orchestration)
- **Chemistry_Traits**: New trait for identifying chemical vs thermal radiation bands, extensible to future photochemistry
- **Operator-split chemical bands**: PE heating and FUV attenuation handled before thermal coupling
- **Numerical improvements**: Line-search damping replaces ad-hoc overshoot clamping; R[n] stored directly (D[n] dropped)
- **Debug tooling**: Compile-time `DiagnosticTrace` for structured Newton iteration diagnostics
- **Dropped**: `beta_order_ >= 2` (future PR), `use_D_as_base`, `SolveLinearEqsWithLastColumn`

### Files
| New | Replaces |
|---|---|
| `coupling_types.hpp` | (new) |
| `opacity_evaluation.hpp` | Part of `source_terms_multi_group.hpp` |
| `dust_closure.hpp` | Part of `radiation_dust_system.hpp` |
| `thermal_solve.hpp` | `source_terms_multi_group.hpp`, `radiation_dust_system.hpp` |
| `flux_update.hpp` | Part of `source_terms_multi_group.hpp` |
| `source_terms.hpp` | `source_terms_single_group.hpp`, `source_terms_multi_group.hpp` |

### Test results
All existing radiation tests pass: RadStreaming, RadhydroShockCGS, RadhydroShockMultigroup, RadMarshakVaytet, RadMarshakAsymptotic, RadhydroUniformAdvecting, RadhydroPulseMGint, RadhydroPulseMGconst, RadMarshakDust, RadMarshakDustPE, RadDust, RadDustMG, RadForce, RadTube, RadhydroBB, ParticleRadiation.
```

- [ ] **Step 2: Commit**

```bash
git add PR.md
git commit -m "docs: add PR description for radiation-matter coupling reimplementation"
```

- [ ] **Step 3: Ask user for confirmation before creating PR**
