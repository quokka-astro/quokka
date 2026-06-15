# SetAtolFromPhysics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `quokka::photochemistry::SetAtolFromPhysics<problem_t>()`, a template function that reads high-level physical parameters (`typical_n_H`, `desired_accuracy_on_T_at_typical_n_H`, `typical_minimal_radiation_T`) from the input `.toml` file and derives VODE absolute tolerances (`atol_spec`, `atol_enuc`, `atol_rad_num`, `atol_rad_flux`, `radiation_failure_tolerance`), injecting them into ParmParse before `init_extern_parameters()` reads them.

**Architecture:** A new header `src/radiation/photochem_atol.H` exports one template function, called from `QuokkaSimulation<problem_t>::Initialize()` between `readParmParse()` (which loads the toml) and `init_extern_parameters()` (which reads `integrator.atol_*` into Microphysics globals). The new high-level parameters and the existing raw `integrator.atol_*` parameters are mutually exclusive; using both triggers `amrex::Abort`. If neither is present the function is a no-op and VODE uses its built-in defaults.

**`Erad_floor` is separate:** `typical_minimal_radiation_T` sets the VODE tolerance; it does not control the M1 radiation floor. `Erad_floor` is a compile-time `constexpr` in each problem's `RadSystem_Traits` specialization and must be set there directly.

**Tech Stack:** C++20, AMReX (`ParmParse`, `Print`), Microphysics `extern_parameters`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/radiation/photochem_atol.H` | **Create** | `SetAtolFromPhysics<problem_t>()` — reads physical params, computes, injects atol values |
| `src/QuokkaSimulation.hpp` | **Modify** (lines 81, 267–272) | Add `#include` and call `SetAtolFromPhysics` before `init_extern_parameters()` |
| `inputs/DTypeFront.toml` | **Modify** | Replace hand-tuned `integrator.atol_*` lines with `integrator.typical_*` lines |

---

## Background: what `init_extern_parameters()` does

The generated file `build/<preset>/src/extern_parameters.cpp` contains:
```cpp
const amrex::ParmParse pp("integrator");
integrator_rp::atol_spec = 1.e-10_rt;          // compiled-in default
pp.query("atol_spec", integrator_rp::atol_spec); // overridden if present in toml
```
`pp.query` reads from the AMReX ParmParse database, populated from the toml at startup.  
`pp.contains("atol_spec")` returns `true` only when the key appears explicitly in the toml.  
`pp.add("atol_spec", value)` inserts (or overwrites) an entry so a subsequent `pp.query` picks it up.

**The injection window:** call `pp.add(...)` after `readParmParse()` (toml loaded) but before `init_extern_parameters()` reads `integrator.atol_*`.

---

## Derivation formulas

| atol variable | Input(s) | Formula |
|---|---|---|
| `atol_spec` | `typical_n_H`, `spec_abundance_tol` (optional, default 1e-5) | `spec_abundance_tol × typical_n_H` |
| `atol_enuc` | `desired_accuracy_on_T_at_typical_n_H` (optional, default 1.0 K) | `(3/2 × k_B / m_p) × desired_accuracy_on_T_at_typical_n_H` |
| `atol_rad_num` | `typical_minimal_radiation_T` | `1e-6 × a_rad × T⁴ / E_photon` |
| `atol_rad_flux` | `rtol_rad_flux` (from toml or default 1e-2) | `= rtol_rad_flux` (forces y_cross = 1; always absolute control) |
| `radiation_failure_tolerance` | derived | `10 × atol_rad_num` |

`typical_minimal_radiation_T` is the minimum physically significant radiation temperature in the problem — the threshold below which photon counts are negligible at 1 ppm of the corresponding blackbody field. It is **not** `Erad_floor`; `Erad_floor` is set independently as a `constexpr` in the problem's `RadSystem_Traits`.

Physical constants from `extern/Microphysics/constants/fundamental_constants.H`:
- `C::a_rad` — radiation constant (erg cm⁻³ K⁻⁴)
- `C::k_B` — Boltzmann constant (erg/K)
- `C::m_p` — proton mass (g)

Photon energy per chem band: `RadSystem<problem_t>::GetChemBandQuanta(0)` — an `AMREX_GPU_HOST_DEVICE` static method, safe to call from host.

---

## Task 1: Create `src/radiation/photochem_atol.H`

**Files:**
- Create: `src/radiation/photochem_atol.H`

- [ ] **Step 1: Write the file**

```cpp
#ifndef PHOTOCHEM_ATOL_H_ // NOLINT
#define PHOTOCHEM_ATOL_H_

#include <cmath>

#include "AMReX.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "fundamental_constants.H"
#include "radiation/radiation_system.hpp"

#ifdef PHOTOCHEMISTRY

namespace quokka::photochemistry
{

namespace detail
{
// Returns true if any integrator.typical_* key appears in the toml
auto hasTypicalParams() -> bool
{
	amrex::ParmParse const pp("integrator");
	return pp.contains("typical_n_H") || pp.contains("typical_minimal_radiation_T") ||
	       pp.contains("desired_accuracy_on_T_at_typical_n_H") || pp.contains("spec_abundance_tol");
}

// Returns true if any integrator.atol_* key appears explicitly in the toml
auto hasRawAtolParams() -> bool
{
	amrex::ParmParse const pp("integrator");
	return pp.contains("atol_spec") || pp.contains("atol_enuc") || pp.contains("atol_rad_num") || pp.contains("atol_rad_flux");
}
} // namespace detail

// SetAtolFromPhysics: derive VODE absolute tolerances from physical scales and inject them
// into ParmParse so init_extern_parameters() picks them up.
//
// Call between readParmParse() and init_extern_parameters() in QuokkaSimulation::Initialize().
// No-op when neither integrator.typical_* nor integrator.atol_* keys appear in the toml.
//
// Note: Erad_floor (the M1 radiation floor) is a compile-time constexpr in RadSystem_Traits
// and is set independently in each problem's specialization. typical_minimal_radiation_T only
// controls the VODE tolerance, not Erad_floor.
template <typename problem_t> void SetAtolFromPhysics()
{
	if (!detail::hasTypicalParams()) {
		return; // backward-compatible: VODE defaults or hand-tuned atol_* take effect
	}
	if (detail::hasRawAtolParams()) {
		amrex::Abort("integrator.typical_* and integrator.atol_* are mutually exclusive. Remove one set.");
	}

	amrex::ParmParse const pp("integrator");

	if (!pp.contains("typical_n_H")) {
		amrex::Abort("integrator.typical_n_H is required when using integrator.typical_* parameters.");
	}
	if (!pp.contains("typical_minimal_radiation_T")) {
		amrex::Abort("integrator.typical_minimal_radiation_T is required when using integrator.typical_* parameters.");
	}

	// --- Read required inputs ---
	amrex::Real typical_n_H = NAN;
	pp.get("typical_n_H", typical_n_H);

	amrex::Real typical_minimal_radiation_T = NAN;
	pp.get("typical_minimal_radiation_T", typical_minimal_radiation_T);

	// --- Read optional inputs ---
	amrex::Real desired_accuracy_on_T_at_typical_n_H = 1.0; // K
	pp.query("desired_accuracy_on_T_at_typical_n_H", desired_accuracy_on_T_at_typical_n_H);

	amrex::Real spec_abundance_tol = 1.0e-5; // fraction of n_H
	pp.query("spec_abundance_tol", spec_abundance_tol);

	amrex::Real rtol_rad_flux = 1.0e-2;
	pp.query("rtol_rad_flux", rtol_rad_flux);

	// --- Physical constants ---
	const amrex::Real E_photon = RadSystem<problem_t>::GetChemBandQuanta(0); // midpoint energy of first chem band (erg)
	const amrex::Real c_v = 1.5 * C::k_B / C::m_p;			       // erg/g/K; monatomic hydrogen

	// --- atol_spec: species negligibility floor ---
	const amrex::Real atol_spec = spec_abundance_tol * typical_n_H;

	// --- atol_enuc: temperature accuracy requirement ---
	const amrex::Real atol_enuc = c_v * desired_accuracy_on_T_at_typical_n_H;

	// --- atol_rad_num: photon number density tolerance ---
	// Set to 1e-6 × the blackbody photon density at typical_minimal_radiation_T.
	// This makes photon counts below 1 ppm of the minimum meaningful radiation field
	// negligible to VODE, so dark cells (Erad ≈ Erad_floor << a*T^4) return in one BDF step.
	// typical_minimal_radiation_T is NOT Erad_floor; Erad_floor is set in RadSystem_Traits.
	const amrex::Real atol_rad_num = 1.0e-6 * C::a_rad * std::pow(typical_minimal_radiation_T, 4.0) / E_photon;

	// --- atol_rad_flux: normalized flux in [0,1] ---
	// Set equal to rtol so y_cross = atol/rtol = 1: always under absolute control.
	const amrex::Real atol_rad_flux = rtol_rad_flux;

	// --- radiation_failure_tolerance ---
	// Must exceed atol_rad_num so a burn is not flagged failed when VODE legitimately
	// steps N_gamma to -atol in a fully-absorbed cell.
	const amrex::Real radiation_failure_tolerance = 10.0 * atol_rad_num;

	// --- Inject into ParmParse (init_extern_parameters reads these) ---
	amrex::ParmParse pp_int("integrator");
	pp_int.add("atol_spec", atol_spec);
	pp_int.add("atol_enuc", atol_enuc);
	pp_int.add("atol_rad_num", atol_rad_num);
	pp_int.add("atol_rad_flux", atol_rad_flux);
	pp_int.add("radiation_failure_tolerance", radiation_failure_tolerance);

	amrex::Print() << "SetAtolFromPhysics: atol_spec=" << atol_spec << " atol_enuc=" << atol_enuc << " atol_rad_num=" << atol_rad_num
	               << " atol_rad_flux=" << atol_rad_flux << " radiation_failure_tolerance=" << radiation_failure_tolerance << "\n";
}

} // namespace quokka::photochemistry
#endif // PHOTOCHEMISTRY
#endif // PHOTOCHEM_ATOL_H_
```

- [ ] **Step 2: Visually verify the file**

Confirm:
- `detail::hasTypicalParams()` lists `typical_n_H`, `typical_minimal_radiation_T`, `desired_accuracy_on_T_at_typical_n_H`, `spec_abundance_tol` — no `typical_luminosity` or `typical_radiation_T`
- `pp.get()` used for the two required fields; `pp.query()` for the three optional ones
- `atol_rad_num` formula is `1e-6 * C::a_rad * T^4 / E_photon` with no `atol_safety_factor`
- No `c_hat` variable (was only needed for the removed `typical_luminosity` path)
- `GetChemBandQuanta(0)` is an `AMREX_GPU_HOST_DEVICE` static — safe to call from host

---

## Task 2: Wire into `QuokkaSimulation<problem_t>::Initialize()`

**Files:**
- Modify: `src/QuokkaSimulation.hpp` at line 81 (include) and lines 267–272 (Initialize body)

- [ ] **Step 3: Add the include immediately after `photochemistry.hpp` (line 82)**

In `src/QuokkaSimulation.hpp`, find:
```cpp
#include "radiation/photochemistry.hpp"
```
Add on the very next line:
```cpp
#include "radiation/photochem_atol.H"
```

- [ ] **Step 4: Add the call between `readParmParse()` and `init_extern_parameters()` (around line 272)**

In `src/QuokkaSimulation.hpp`, find this exact block:
```cpp
		// read in runtime parameters
		readParmParse();
		// set gamma
		amrex::ParmParse eos("eos");
		eos.add("eos_gamma", quokka::EOS_Traits<problem_t>::gamma);
		// initialize Microphysics params
		init_extern_parameters();
```
Replace with:
```cpp
		// read in runtime parameters
		readParmParse();
		// set gamma
		amrex::ParmParse eos("eos");
		eos.add("eos_gamma", quokka::EOS_Traits<problem_t>::gamma);
#ifdef PHOTOCHEMISTRY
		// if integrator.typical_* keys are present, derive atol values and inject them into
		// ParmParse before init_extern_parameters() reads integrator.atol_*
		quokka::photochemistry::SetAtolFromPhysics<problem_t>();
#endif
		// initialize Microphysics params
		init_extern_parameters();
```

---

## Task 3: Update `inputs/DTypeFront.toml`

**Files:**
- Modify: `inputs/DTypeFront.toml`

DTypeFront has `Erad_floor = C::a_rad * 1.0e-4` (T=0.1K) set in `testDTypeFront.cpp`. That remains unchanged. `typical_minimal_radiation_T` below is the minimum physically meaningful radiation temperature for tolerance purposes, not the floor temperature.

For DTypeFront, 100 K gives:
- `atol_rad_num = 1e-6 × a_rad × (100)^4 / E_photon ≈ 3.5e-2 cm⁻³`
- This is >> `N_gamma_floor = a_rad × (0.1K)^4 / E_photon ≈ 3.5e-8 cm⁻³`, so dark cells return in one BDF step ✓

- [ ] **Step 5: Replace the hand-tuned `atol_*` block**

In `inputs/DTypeFront.toml`, find and replace this block:
```toml
integrator.atol_spec = 1e-3   # cm^-3; 1e-4 × n_H = physical negligibility floor
integrator.rtol_spec = 1e-2
integrator.atol_rad_num = 1e-2  # cm^-3; must exceed N_gamma_floor = Erad_floor/E_photon ~ 3.5e-8 so dark cells return in one VODE step
integrator.radiation_failure_tolerance = 0.1  # N_gamma can dip to -atol_rad_num = -0.01 in cells where photons are fully absorbed; must exceed atol_rad_num
integrator.rtol_rad_num = 1e-2
integrator.atol_rad_flux = 1e-2  # normalized flux (dimensionless); crossover y_cross = atol/rtol = 1, always absolute control
integrator.rtol_rad_flux = 1e-2
integrator.atol_enuc = 1.24e8  # erg/g; ~2.5% of e at T=37 K neutral gas
integrator.rtol_enuc = 1e-2
```
With:
```toml
# Physical scales: SetAtolFromPhysics() derives all integrator.atol_* values from these.
# Erad_floor (the M1 floor) is set separately in testDTypeFront.cpp as a_rad*(0.1K)^4.
integrator.typical_n_H                      = 139.9  # cm^-3; representative total H number density
integrator.desired_accuracy_on_T_at_typical_n_H = 1.0   # K; temperature accuracy (sets atol_enuc)
integrator.typical_minimal_radiation_T      = 100.0  # K; minimum physically significant radiation T

integrator.rtol_spec     = 1e-2
integrator.rtol_rad_num  = 1e-2
integrator.rtol_rad_flux = 1e-2
integrator.rtol_enuc     = 1e-2
```

---

## Task 4: Build and validate

**Files:**
- Build artifact: `build/3d/src/problems/DTypeFront/DTypeFront`

- [ ] **Step 6: Build**

```bash
quokka build -d 3d DTypeFront --root /home/ubuntu/workspace
```

Expected: build succeeds with no errors.

**If build fails** with "calling `__device__` function from `__host__`" on `GetChemBandQuanta`:  
Replace the call with the equivalent inline computation:
```cpp
// Alternative if GetChemBandQuanta is not callable from host:
auto const freq_bounds = RadSystem_Traits<problem_t>::ChemBands();
const amrex::Real E_photon = 0.5 * (static_cast<amrex::Real>(freq_bounds[0]) +
                                     static_cast<amrex::Real>(freq_bounds[1])) * C::hplanck;
```

- [ ] **Step 7: Clean and run**

```bash
quokka clean --root /home/ubuntu/workspace
quokka run -d 3d DTypeFront --root /home/ubuntu/workspace 2>&1 | grep -E "SetAtolFromPhysics|Test passed|Test FAILED|figure-of-merit|evolve\(\)|HydroSolver"
```

Expected output (values are approximate):
```
SetAtolFromPhysics: atol_spec=0.001399 atol_enuc=1.2396e+08 atol_rad_num=3.47e-02 atol_rad_flux=0.01 radiation_failure_tolerance=0.347
Test passed: D-type front effective radius matches analytical radius within 2.598076211 cell sizes at end of simulation.
Test passed: cavity median temperature ... is within 5% of analytical equilibrium ...
Test passed: neutral median temperature ... is within 5% of analytical equilibrium ...
DTypeFront SUCCESS
```

**If the neutral temperature test fails:**  
Check the printed `atol_rad_num`. It must exceed `N_gamma_floor = a_rad × (0.1K)^4 / E_photon ≈ 3.5e-8 cm⁻³`. With `typical_minimal_radiation_T = 100K` the derived value is ≈ 3.47e-2 >> 3.5e-8 ✓. If `E_photon` is wrong (not ≈ 2.18e-11 erg), check that `GetChemBandQuanta(0)` returns the H Lyman-edge photon energy.

- [ ] **Step 8: Check performance ratio**

```bash
quokka run -d 3d DTypeFront --root /home/ubuntu/workspace 2>&1 | grep -E "AMRSimulation::evolve|REG::HydroSolver" | head -4
```

Expected: `AMRSimulation::evolve()` / `REG::HydroSolver` ratio ≤ 15× (hand-tuned baseline was 8.5×; `atol_rad_num = 3.47e-2` vs hand-tuned `1e-2` is slightly less aggressive but dark cells remain under absolute control).

If ratio exceeds 20×, `atol_rad_num` is too small — raise `typical_minimal_radiation_T` in the toml and re-run.

- [ ] **Step 9: Commit**

```bash
git add src/radiation/photochem_atol.H src/QuokkaSimulation.hpp inputs/DTypeFront.toml
git commit -m "$(cat <<'EOF'
feat(photochem): add SetAtolFromPhysics to derive VODE atol from physical parameters

Users now specify integrator.typical_n_H, integrator.desired_accuracy_on_T_at_typical_n_H,
and integrator.typical_minimal_radiation_T instead of hand-tuning integrator.atol_spec /
atol_enuc / atol_rad_num / atol_rad_flux.

SetAtolFromPhysics<problem_t>() injects derived values into ParmParse before
init_extern_parameters() reads them, leaving the Microphysics submodule untouched.
The new keys and existing integrator.atol_* keys are mutually exclusive.
Neither present is a backward-compatible no-op.

Erad_floor (the M1 radiation floor) remains a compile-time constexpr in each
problem's RadSystem_Traits; typical_minimal_radiation_T only sets the VODE
tolerance, not Erad_floor.

DTypeFront.toml updated to demonstrate the interface.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- [x] `SetAtolFromPhysics<problem_t>()` template in new header — Task 1
- [x] ParmParse re-injection pattern — Task 1 (`pp_int.add(...)`)
- [x] `typical_n_H` → `atol_spec` — Task 1
- [x] `desired_accuracy_on_T_at_typical_n_H` → `atol_enuc` — Task 1
- [x] `typical_minimal_radiation_T` → `atol_rad_num = 1e-6 × a_rad × T⁴ / E_photon` — Task 1
- [x] `typical_luminosity` removed — Task 1 (not present)
- [x] `atol_rad_flux = rtol_rad_flux` — Task 1
- [x] `radiation_failure_tolerance = 10 × atol_rad_num` — Task 1
- [x] `Erad_floor` is separate (constexpr in problem generator) — noted in header comment and toml comment
- [x] Conflict detection (`typical_*` + `atol_*` both present) — Task 1
- [x] Missing required param detection — Task 1
- [x] No-op when neither group present — Task 1
- [x] Call site in `QuokkaSimulation::Initialize()` — Task 2
- [x] DTypeFront.toml updated — Task 3
- [x] Build + test validation — Task 4

**Placeholder scan:** No TBDs or "implement later" language found.

**Type consistency:**
- `SetAtolFromPhysics<problem_t>()` — same void signature in Tasks 1, 2, and commit ✓
- `RadSystem<problem_t>::GetChemBandQuanta(0)` — static `AMREX_GPU_HOST_DEVICE` returning `amrex::Real` ✓
- `C::a_rad`, `C::k_B`, `C::m_p` — namespace `C`, from `fundamental_constants.H` ✓
- `pp.add(const char*, amrex::Real)` — standard AMReX ParmParse API ✓
- `detail::` namespace — prevents ODR issues across TUs ✓
- `desired_accuracy_on_T_at_typical_n_H` — consistent across `hasTypicalParams()`, `pp.query()`, and the toml ✓
- `typical_minimal_radiation_T` — consistent across `hasTypicalParams()`, `pp.get()`, and the toml ✓
