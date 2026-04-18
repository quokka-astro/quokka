# ADR-0001: Unify Gas Temperature Through EOS
Date: 2026-04-18 • Status: Proposed

## Context
Quokka currently has two independent runtime ways to compute gas temperature from density and internal energy:

- `quokka::EOS<problem_t>::ComputeTgasFromEint(...)` in `src/hydro/EOS.hpp`.
  For non-chemistry hydro problems this is the gamma-law EOS path with a constant `mean_molecular_weight`.
- `quokka::ResampledCooling::ComputeTgasFromEgas(...)` in `src/cooling/ResampledCooling.hpp`.
  This interpolates temperature directly from the resampled cooling tables on the `(rho, e_int)` grid.

When `cooling.cooling_table_type = resampled`, the table-backed value is currently treated as the authoritative temperature in several problem specializations, because it matches the cooling-table thermodynamics while the generic EOS still assumes a fixed mean-molecular-weight ideal-gas conversion.

This produces a split-brain (and inconsistent) design:

- Core hydro and radiation machinery uses `EOS`:
  `src/hydro/hydro_system.hpp`, `src/hydro/NSCBC_inflow.hpp`, `src/radiation/source_terms_single_group.hpp`, `src/radiation/source_terms_multi_group.hpp`, `src/radiation/radiation_dust_system.hpp`, and `src/radiation/radiation_system.hpp`.
- Resampled-cooling diagnostics and temperature-threshold logic bypass `EOS` and call the cooling table directly:
  `src/problems/DiskGalaxy/testDiskGalaxy.cpp`,
  `src/problems/TallBoxSf/testTallBoxSf.cpp`,
  `src/problems/ShockCloud/testShockCloud.cpp`,
  `src/problems/RandomBlast/testRandomBlast.cpp`,
  and `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp`.

As a result, the same state can have:

- one temperature used by diagnostics, statistics, and thresholding when they call the resampled table directly; and
- another temperature implied by `EOS::ComputeTgasFromEint(...)`, temperature floors, and any other generic machinery that routes through `EOS`.

This inconsistency is a blocker for new physics modules that depend explicitly on gas temperature and are expected to work with `ResampledCooling`. Planned physics modules such as thermal conduction and photoionization need one authoritative temperature value to correctly compute their source terms.

### Code Audit Summary

What is already consistent:

- Generic hydro/radiation source terms already go through `EOS`.
- Problem-specific analytical EOS specializations used by test problems (for example Su-Olson or Marshak variants) still live inside the `EOS` interface and are not part of this inconsistency.

What is independently computed outside the runtime EOS path:

- Resampled-cooling problem diagnostics and statistics listed above.
- Test/post-processing formulas:
  `src/problems/RadTube/testRadTube.cpp` computes temperature from pressure during setup,
  `src/problems/SN/testSN.cpp` computes temperature from `Eint / (rho * C_V)` for plotting,
  and `scripts/python/quick_plot` falls back to an ideal-gas temperature estimate when a plotfile does not contain a temperature field.

## Options
- Option A: Keep the status quo.
  Continue using `EOS` for generic code and direct resampled-table interpolation in selected problem specializations.

- Option B: Patch individual call sites to use the resampled cooling helper where needed.
  This would make more places numerically correct for resampled cooling, but it would further fragment the temperature API and leave generic `EOS` users inconsistent.

- Option C: Make `EOS` the single authority for gas temperature, with a runtime-selectable resampled-cooling backend.
  `EOS` remains the public interface. When resampled cooling is active, `EOS` dispatches temperature-related calls through the cooling-table thermodynamics instead of the fixed-`mu` ideal-gas conversion.

- Option D: Replace the full hydro thermodynamic closure with the resampled tables.
  Route temperature, pressure, sound speed, and entropy through a unified table-backed EOS everywhere.
  This is broader than the current problem statement and would change the hydro closure, wave speeds, and likely solver behavior.

## Decision
Choose Option C.

Quokka should keep a single public thermodynamics interface in `EOS`, and temperature-related quantities should be routed through that interface in all runtime code paths.

When `cooling.cooling_table_type = resampled`, `EOS` should return the same temperature that is currently obtained only from `ResampledCooling::ComputeTgasFromEgas(...)`.

The proposed design is:

- Keep `quokka::EOS<problem_t>` as the only public runtime API for gas temperature.
- Add a temperature backend selection inside `EOS`:
  default ideal-gas/chemistry behavior remains unchanged;
  a resampled-cooling backend is activated when the simulation loads resampled cooling tables.
- Register a GPU-usable resampled-table handle with the EOS runtime during simulation setup, after `readResampledData(...)`.
- Implement these `EOS` functions against the active backend:
  `ComputeTgasFromEint(rho, Eint, ...)`
  `ComputeEintFromTgas(rho, Tgas, ...)`
  `ComputeEintTempDerivative(rho, Tgas, ...)`
- For the resampled backend:
  `ComputeTgasFromEint(...)` interpolates the existing `T(rho, e_int)` table.
  `ComputeEintFromTgas(...)` uses an inverse relation built from the same data.
  `ComputeEintTempDerivative(...)` is derived from that same inverse relation so the EOS API remains self-consistent.
- Migrate all resampled-cooling runtime consumers that currently call `ResampledCooling::ComputeTgasFromEgas(...)` to call `EOS::ComputeTgasFromEint(...)` instead.
- Treat `ResampledCooling::ComputeTgasFromEgas(...)` as an internal helper for the EOS backend and transition plan, not as a public runtime interface for general problem code.


## Rollout & Testing
1. Add an EOS temperature-backend mechanism with a resampled-cooling registration step in `QuokkaSimulation`.
2. Build or load an inverse `e_int(rho, T)` representation from the same resampled cooling data used for `T(rho, e_int)`.
3. Implement `ComputeTgasFromEint`, `ComputeEintFromTgas`, and `ComputeEintTempDerivative` against that backend.
4. Replace direct runtime calls to `ResampledCooling::ComputeTgasFromEgas(...)` in:
   `DiskGalaxy`, `TallBoxSf`, `ShockCloud`, `RandomBlast`, and `ResampledCoolingTest`.
5. Add targeted regression coverage:
   CPU and at least one GPU backend.

Recommended tests:

- EOS/table agreement test:
  sample `(rho, Eint)` points across the resampled table domain and verify that `EOS::ComputeTgasFromEint(...)` matches the current table interpolation within interpolation tolerance.
- Round-trip thermodynamics test:
  verify `Eint -> T -> Eint` using the EOS backend over the table domain.
- Temperature floor test:
  verify that enforcing `temperature_floor` under resampled cooling yields the expected post-floor temperature from the table-backed EOS.
- Problem regression:
  compare one resampled-cooling problem's derived `temperature` output before and after the refactor and require bitwise agreement or a clearly justified interpolation tolerance.

## Links
- `src/hydro/EOS.hpp`
- `src/cooling/ResampledCooling.hpp`
- `src/hydro/hydro_system.hpp`
- `src/problems/DiskGalaxy/testDiskGalaxy.cpp`
- `src/problems/TallBoxSf/testTallBoxSf.cpp`
- `src/problems/ShockCloud/testShockCloud.cpp`
- `src/problems/RandomBlast/testRandomBlast.cpp`
- `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp`
