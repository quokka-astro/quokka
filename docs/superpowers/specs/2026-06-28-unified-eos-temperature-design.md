# Design: Unified EOS infrastructure for gas temperature

Date: 2026-06-28
Status: Approved (design phase)
Supersedes the narrow framing in `ADR-0001-unify-gas-temperature-through-eos.md`; tracks issue #1810 and PR #2012.
Builds on merged PR #2008 (device-memory `DataTable` with host/device split).

## Problem

Quokka has two independent runtime paths that compute gas temperature from
`(rho, Eint)`:

- `quokka::EOS<problem_t>::ComputeTgasFromEint(...)` — the gamma-law (or, under
  `CHEMISTRY`/`PHOTOCHEMISTRY`, the Microphysics) path.
- `quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, tables)` — direct
  interpolation of the resampled cooling `T(rho, e_int)` table.

When `cooling.cooling_table_type = resampled`, the table value is the physically
correct temperature, but only diagnostics and threshold logic call it directly
(17 call sites across DiskGalaxy, TallBoxSf, ShockCloud, RandomBlast,
ResampledCoolingTest, and ElectronConduction). Generic hydro/radiation machinery
routes through `EOS`, which still returns the fixed-`mu` ideal-gas temperature.
The same cell therefore has two temperatures. This split-brain blocks new physics
modules (thermal conduction, photoionization) that need one authoritative `T`.

## Decision summary

Make `quokka::EOS<problem_t>` the single authority for gas temperature by
introducing a **compile-time EOS backend hierarchy** selected per problem via a
trait, defaulting to the existing gamma-law behavior so the vast majority of
problems are untouched. A new `EOSTabulated` backend routes the temperature
functions through the resampled table. Table data reaches the static, GPU-callable
EOS methods through a process-global registry populated once at simulation setup.

Key decisions (from design dialogue):

1. **Full class hierarchy** (compile-time/trait polymorphism, not the narrower
   ADR Option C in-place runtime switch, and not runtime virtual dispatch which is
   GPU-incompatible).
2. **Registered global handle** delivers runtime table data to the static EOS
   methods — near-zero call-site churn.
3. **Temperature only**: the tabulated backend serves `T` quantities from the
   table; pressure and sound speed stay on the ideal-gas formula so the hydro
   Riemann closure / wave speeds are unchanged. (Option D is out of scope.)
4. **Strict compile-time selection**: a problem compiled with `EOSTabulated`
   requires registered tables at runtime and aborts otherwise.
5. **Gamma-law default trait** (à la `DefaultPhysicsTraits`): existing problems
   inherit the default and need no changes; only the 5 resampled-cooling problems
   opt in.

## 1. Architecture — compile-time backend hierarchy

A set of backend structs, each implementing the same static `HOST_DEVICE`
interface (the current `EOS` method set). There is **no virtual dispatch** —
"hierarchy" means compile-time/trait polymorphism, the form converged on in #1810.

- `EOSIdeal<problem_t>` — the current gamma-law `#else` path (uses Microphysics
  `chem_eos_t` gamma-law `eos()`); depends on `eos.H` but not on the reaction
  network.
- `EOSMicrophysics<problem_t>` — the current `#if defined(CHEMISTRY) ||
  defined(PHOTOCHEMISTRY)` path; only compiled and only selectable when those
  macros are defined (Microphysics network headers present).
- `EOSTabulated<problem_t>` — temperature methods (`ComputeTgasFromEint`,
  `ComputeEintFromTgas`, `ComputeEintTempDerivative`) read the resampled table;
  all other methods (pressure, sound speed, isothermal sound speed, pressure
  derivatives, `ComputeEintFromPres`, `ComputeOtherDerivatives`) delegate to
  `EOSIdeal` (temperature-only scope).

`quokka::EOS<problem_t>` remains the single public name. It forwards to / inherits
from the trait-selected backend and retains the backend-independent kinematic
helpers (`ComputeEintFromEgas`, `ComputeEgasFromEint`) and the `gamma_` /
`boltzmann_constant_` static members. **Every existing consumer call site is
unchanged** — they keep calling `quokka::EOS<problem_t>::Compute...(...)`.

## 2. Backend selection — trait with gamma-law default

Following the `DefaultPhysicsTraits` pattern, add a `backend` selector to the EOS
traits with a default that resolves to:

- `EOSMicrophysics` when `CHEMISTRY`/`PHOTOCHEMISTRY` is defined, else
- `EOSIdeal`.

This reproduces today's behavior exactly for all current problems, which inherit
the default and remain untouched. The 5 resampled-cooling problems
(DiskGalaxy, TallBoxSf, ShockCloud, RandomBlast, ResampledCoolingTest) specialize
the trait to `EOSTabulated`.

Selection is **strict**: an `EOSTabulated`-selected problem requires registered
tables at runtime. A tabulated temperature call before/without registration hits a
device-safe assertion (`AMREX_ALWAYS_ASSERT_WITH_MESSAGE` / device `printf` +
abort path), never reads uninitialized memory.

## 3. Runtime table delivery — registered global handle

A translation-unit-scope registry, allocated in **managed memory**
(`The_Managed_Arena`) so a `HOST_DEVICE` EOS method can read it from either side:

```
struct EOSTabulatedRegistry {
    bool active = false;
    quokka::ResampledCooling::resampledGpuConstTables host;    // pinned-host pointers
    quokka::ResampledCooling::resampledGpuConstTables device;  // device pointers
};
```

- `device` is built from `resampled_tables::const_tables()` → internal `Table1D`
  pointers are pure **device** memory (PR #2008 `The_Device_Arena`); dereferenced
  **only** inside `AMREX_IF_ON_DEVICE`.
- `host` is built from `resampled_tables::const_tables_host()` → pinned host
  pointers; dereferenced **only** inside `AMREX_IF_ON_HOST`.
- The registry **container** is managed memory; the sub-handle pointers are each
  consumed only in their matching execution context.

**Registration point**: immediately after the existing `readResampledData(...)`
call in `QuokkaSimulation.hpp` (~line 666). `sync_tables_to_device()` already runs
inside the `DataTable` initialization path (DataTable.hpp:945/987), so both
`const_tables()` and `const_tables_host()` are valid the moment
`readResampledData(...)` returns — no additional sync step is required. Registration
sets `active = true`.

`EOSTabulated`'s temperature methods read the registry, assert `active`, then pick
`device` or `host` via `AMREX_IF_ON_DEVICE` / `AMREX_IF_ON_HOST`. No EOS method
signature changes.

## 4. Inverse relations for the tabulated backend

- `ComputeTgasFromEint(rho, Eint)` → interpolate the `T(rho, e_int)` table
  (equivalent to today's `ResampledCooling::ComputeTgasFromEgas`).
- `ComputeEintFromTgas(rho, Tgas)` → reuse the existing
  `ResampledCooling::ComputeEgasFromTgas` (monotone root-find via toms748).
- `ComputeEintTempDerivative(rho, Tgas)` (`dEint/dT`) → derived from the same
  inverse relation (finite difference of `Eint(T)` about `Tgas`), keeping the EOS
  API self-consistent.

`ResampledCooling::ComputeTgasFromEgas` / `ComputeEgasFromTgas` become internal
helpers backing `EOSTabulated`, not a public runtime temperature interface.

## 5. Migration scope (deliberately small)

- **Generic consumers** (`hydro_system.hpp`, `radiation/*`, `NSCBC_inflow.hpp`):
  **untouched**. They already call `EOS<problem_t>::ComputeTgasFromEint`; for the 5
  tabulated problems they now transparently receive the table temperature.
- **The 5 problems' diagnostics** (17 direct `ResampledCooling::ComputeTgasFromEgas`
  call sites): migrate to `EOS<problem_t>::ComputeTgasFromEint`. Localized; removes
  the fragmentation. Host-side sites continue to resolve through the EOS host path
  (registry host sub-handle), reading the same pinned memory as today.
- **`ElectronConduction`**: remove the **temperature** branch of the
  `EOSFlagforConduction` enum — always call `EOS::ComputeTgasFromEint`. **Sound
  speed** stays a direct `ResampledCooling::ComputeSoundSpeedFromRhoEint` call (cs
  is out of scope under temperature-only). This is the single documented remaining
  seam.

## 6. Scope boundaries (out of scope)

- Pressure / sound speed / entropy unification through the table (Option D) —
  would change the hydro closure and wave speeds.
- The EOS ↔ VODE / Microphysics fork discussion and integrator replacement.
- Per-problem custom mixed EOS (e.g. photoionization low-T microphysics + high-T
  tabulated); the hierarchy leaves room for it later but it is not specified here.

## 7. Testing

- **EOS/table agreement**: sample `(rho, Eint)` across the table domain;
  `EOS::ComputeTgasFromEint` matches raw table interpolation within interpolation
  tolerance.
- **Round-trip thermodynamics**: `Eint → T → Eint` via the tabulated backend over
  the domain.
- **Temperature floor**: enforcing `temperature_floor` under resampled cooling
  yields the expected post-floor temperature from the table-backed EOS.
- **Problem regression**: one resampled-cooling problem's derived `temperature`
  output, before vs. after — require **bitwise** agreement (the migration reads the
  same tables, so it should be value-preserving), or a clearly justified
  interpolation tolerance.
- Coverage on **CPU and at least one GPU backend**.

## 8. Risks and mitigations

- **Strict selection requires tables in every CI input** for the 5 tabulated
  problems, else they abort. All 5 use `cooling_table_type = "resampled"` today;
  the implementation plan must explicitly verify each problem's CI input(s),
  DiskGalaxy in particular (it did not appear in the resampled-input grep).
- **Registry init ordering**: the registry must read `active = false` until
  registration; any tabulated EOS call before registration must hit the
  device-safe assertion, not read garbage. Verify no tabulated problem invokes a
  temperature method during setup before `readResampledData(...)`.
- **Host/device pointer mismatch**: device sub-handle dereferenced on host (or
  vice versa) is undefined behavior. The `AMREX_IF_ON_DEVICE` / `AMREX_IF_ON_HOST`
  split is the guard; covered by the CPU + GPU agreement tests.

## Affected files (anticipated)

- `src/hydro/EOS.hpp` — backend structs, `EOS<problem_t>` selector, trait default.
- `src/cooling/ResampledCooling.hpp` / `.cpp` — registry definition + registration
  helper; demote `ComputeTgasFromEgas`/`ComputeEgasFromTgas` to internal helpers.
- `src/QuokkaSimulation.hpp` — register the global handle after
  `readResampledData(...)`.
- `src/conduction/ElectronConduction.hpp` — drop the temperature branch of
  `EOSFlagforConduction`.
- The 5 problem files — trait specialization to `EOSTabulated` + migrate the 17
  diagnostic call sites.
- New regression test target(s) for EOS/table agreement, round-trip, and floor.
