# Cooling Module

Radiative cooling in Quokka is selected at compile time by the `EOSBackend` member of `quokka::EOS_Traits<problem_t>`. The backend determines both how gas temperature is computed and whether a cooling source term is applied at all — there is no separate "enable cooling" trait.

## Choosing an EOS backend

| Backend | Cooling | Temperature from |
|---------|---------|------------------|
| `quokka::EOSIdeal<P>` | none | ideal gas law with constant \\(\mu\\) |
| `quokka::EOSTabulated<P>` | tabulated (Grackle-derived) | resampled table lookup |
| `quokka::EOSMicrophysics<P>` | Microphysics reaction network | Microphysics EOS |

A problem selects its backend in its `EOS_Traits` specialization:

```cpp
template <> struct quokka::EOS_Traits<MyProblem> {
    static constexpr double gamma = 5. / 3.;
    static constexpr double mean_molecular_weight = C::m_u;
    using EOSBackend = quokka::EOSTabulated<MyProblem>;
};
```

If `EOSBackend` is omitted, the default is `EOSMicrophysics<P>` when the target is compiled with `CHEMISTRY` or `PHOTOCHEMISTRY` defined, and `EOSIdeal<P>` otherwise.

### EOSIdeal

The gamma-law ideal gas EOS, and the default for most problems. Temperature follows from \\(e\_{\rm int} = k\_B T / [(\gamma - 1) \mu m\_u]\\) with the constant `mean_molecular_weight` from `EOS_Traits`. No cooling source term is applied, and `cooling.*` parameters have no effect.

### EOSTabulated

Applies optically-thin radiative cooling as a Strang-split source term, and computes temperature by interpolating a pre-computed table of \\((\rho, e\_{\rm int})\\). This is the backend used by the ISM and galaxy problems (`ShockCloud`, `TallBoxSf`, `DiskGalaxy`, `RandomBlast`, `SN`, `ParticleSF`). It is described in detail below.

### EOSMicrophysics

Delegates the EOS to the [Microphysics](https://github.com/AMReX-Astro/Microphysics) submodule, so that temperature is consistent with the evolving chemical species tracked by the reaction network (`chemstate.xn`). Cooling and heating are handled by the network itself rather than by a table, and are advanced by the chemistry burner (`quokka::chemistry::computeChemistry`) rather than by the cooling integrator. This backend is only used for primordial (Pop III) chemistry problems such as `PrimordialChem`, whose CMake target compiles with `-DCHEMISTRY`; see the chemistry module for details.

## The EOSTabulated cooling model

Each hydrodynamic timestep, the cooling operator integrates the internal energy of every cell at fixed density (isochoric cooling):

<script type="math/tex; mode=display">
\frac{d E_{\rm int}}{d t} = \dot{E}_{\rm tab}(\rho, e_{\rm int}) \, \rho^2 + \Gamma \, n_H
</script>

The first term is interpolated from the table; the second is a spatially uniform heating rate per hydrogen atom (erg s⁻¹ H⁻¹), where \\(n\_H = \rho \, X\_H / m\_p\\) and \\(X\_H\\) is the Cloudy hydrogen mass fraction stored in the table. \\(\Gamma\\) is the sum of the [SFH-based photoelectric heating rate](#star-formation-history-based-photoelectric-heating) and the `heating_rate_external` parser expression; both are zero by default.

The tabulated term \\(\dot{E}\_{\rm tab}\\) bundles three physical processes, all baked into the table when it is generated:

1. **Grackle/Cloudy heating and cooling** — primordial and metal heating and cooling rates, plus UV background photo-heating, read from the Grackle data files. Metal rates are tabulated at solar metallicity and scaled linearly by the metallicity chosen at generation time (`--zmet`).
2. **Photoelectric heating** — grain photoelectric heating using the \\(\epsilon(G\_0, T, n\_e, \phi\_{\rm PAH})\\) prescription of Wolfire et al. (2003), with \\(G\_0 = 1.7\\) and \\(\phi\_{\rm PAH} = 0.5\\), scaled by metallicity. Included only in tables generated without `--exclude_pe`.
3. **Inverse Compton cooling off CMB photons** — \\(\propto -(T - T\_{\rm CMB}) \, n\_e\\) with \\(T\_{\rm CMB} = 2.725\\) K, following e.g. Hirata (2018). Always included.

The ODE is integrated with an adaptive embedded RK2 (Heun's method) sub-cycle at a relative tolerance of \\(10^{-4}\\). If `temperature_floor` is set, the result is clamped to the corresponding internal energy and the floor also sets the absolute integration tolerance. If a cell exceeds the maximum substep count, the cooling operator reports failure and the hydro update is retried.

## Cooling tables

Each table covers a 2D grid of \\((\rho, e\_{\rm int})\\) with `fast_log` coordinate spacing (a bit-level approximation to log₂) and stores five outputs as raw physical values:

| Index | Quantity | Units |
|-------|----------|-------|
| 0 | Cooling rate \\(\dot{E}\_{\rm tab} = \dot{E}/\rho^2\\) | erg cm³ g⁻² s⁻¹ |
| 1 | Temperature | K |
| 2 | Sound speed | cm s⁻¹ |
| 3 | Pressure | dyn cm⁻² |
| 4 | Entropy \\(K = k\_B T \, n^{-2/3}\\) | erg cm² |

Available table files in `extern/cooling/`:

| File | UV background | Photoelectric heating |
|------|---------------|-----------------------|
| `CloudyData_UVB=HM2012_resampled.h5` | HM2012 | included |
| `CloudyData_UVB=HM2012_resampled_noPE.h5` | HM2012 | excluded |
| `CloudyData_UVB=HM2012_shielded_resampled.h5` | HM2012 (shielded) | included |
| `CloudyData_UVB=HM2012_shielded_resampled_noPE.h5` | HM2012 (shielded) | excluded |
| `isrf_1000Go_grains_resampled.h5` | ISRF 1000 G₀ + grains | included |

All bundled tables are generated at solar metallicity and redshift zero. To regenerate the four Grackle-based tables, run `extern/cooling/resample_grackle_cooling_tables.sh`; to regenerate `isrf_1000Go_grains_resampled.h5`, run `extern/cooling/resample_cloudy_cooling_tables.py` directly.

### Source Grackle data files

The Grackle-based tables are resampled from the Cloudy data files distributed with [Grackle](https://grackle.readthedocs.io/en/latest/Parameters.html), which are downloaded automatically by `resample_grackle_cooling_tables.py`. These files store heating and cooling rates for primordial and metal species along with UV background photo-heating and photo-ionization rates, valid over \\(-10 < \log\_{10}(n\_H / {\rm cm^{-3}}) < 4\\) and \\(1 < \log\_{10}(T / {\rm K}) < 9\\); Grackle extrapolates outside this range. Quokka uses two of them:

- `CloudyData_UVB=HM2012.h5` — rates with the [Haardt & Madau (2012)](http://adsabs.harvard.edu/abs/2012ApJ...746..125H) UV background. Collisional ionization equilibrium is assumed above redshift 15.13.
- `CloudyData_UVB=HM2012_shielded.h5` — the same UV background, but recomputed with Jeans-length depth models (capped at 100 pc) so that metal line cooling is correct in self-shielded regions. Using the optically-thin table where self-shielding matters can overestimate the net cooling rate by an order of magnitude at some densities.

**Note:** the optically-thin HM2012 file is known to have unphysically high heating rates at low densities ([grackle_data_files issue #7](https://github.com/grackle-project/grackle_data_files/issues/7)). Grackle recommends against it for fully tabulated cooling; prefer the shielded table for ISM problems.

**Note:** Output transform metadata is intentionally absent from the HDF5 files. Output transforms are a property of the interpolation, not of the data; they are declared at the C++ call site in `ResampledCooling.cpp`. HDF5 files always contain raw physical values. When `H5Reader` loads a `fast_log` output, it applies `FastMath::inverse_pow2` (Newton iteration) to each value at load time, so the internal buffer stores the interpolation-space representation that `FastMath::pow2` inverts during lookup.

## HDF5 table format

All cooling tables follow the self-describing `tab1` group format read by `quokka::DataTable`. The group `tab1` inside each HDF5 file contains:

- **Dataset** `data` — shape `[5, Nx0, Nx1]` (C row-major), holding the five output quantities stacked along the first axis.
- **Dataset** `grids/rho` — physical density grid values (g/cm³), informational only.
- **Dataset** `grids/eint` — physical specific internal energy grid values (erg/g), informational only.
- **Attributes** on `tab1`:

  | Attribute | Type | Description |
  |-----------|------|-------------|
  | `Ndim` | int32 | Number of input dimensions (2) |
  | `Nout` | int32 | Number of output quantities (5) |
  | `Nx` | int32[2] | Grid sizes `[n_rho, n_eint]` |
  | `xlo` | float64[2] | Physical lower bounds `[rho_min, eint_min]` |
  | `xhi` | float64[2] | Physical upper bounds `[rho_max, eint_max]` |
  | `spacing` | string[2] | Coordinate spacing type (`"fast_log"` for both) |
  | `input_names` | string[2] | `["rho", "eint"]` |
  | `output_names` | string[5] | Names for the five outputs |
  | `input_units` | string[2] | Physical units of inputs |
  | `output_units` | string[5] | Physical units of outputs |
  | `include_pe` | int32 | 1 if photoelectric heating is included, else 0 |
  | `cloudy_H_mass_fraction` | float64 | Hydrogen mass fraction assumed in Cloudy |

  String attributes must use fixed-length HDF5 strings (numpy `dtype='S'`); variable-length HDF5 strings are not supported by `H5Reader`.

Old-format files (pre-`tab1`) can be converted in-place with `scripts/python/convert_cooling_table_hdf5.py`.

## Runtime parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cooling.enabled` | bool (0/1) | `1` | Only takes effect with the `EOSTabulated` backend. Set to 0 to skip the cooling integrator while still using the table for temperature — useful for testing. |
| `cooling.hdf5_data_file` | string | **required** with `EOSTabulated` | Path to the cooling table HDF5 file. |
| `cooling.cooling_table_type` | string | `"resampled"` | Table type. Only `"resampled"` is supported. |
| `cooling.read_tables_even_if_disabled` | bool (0/1) | `0` | Read tables at startup even when the problem does not use `EOSTabulated` (useful for diagnostics). |
| `heating_rate_external` | string | `""` | AMReX parser expression for a time-variable external heating rate per H atom (erg s⁻¹ H⁻¹). Variables: `time`, `dt`; constants: `yr`, `kyr`, `Myr`, `Gyr`. |
| `temperature_floor` | float | `0.0` | Minimum temperature (K). Clamps the cooling integration and sets its absolute tolerance. |

See [Runtime parameters](parameters.md) for the full parameter table, including the star formation history parameters below.

## Star-formation-history-based photoelectric heating

By default, photoelectric heating is a static part of the table, computed for a fixed \\(G\_0 = 1.7\\). For problems where the ISRF should instead track the star formation history of the simulation itself, set `use_sfh_based_pe_heating = 1`. Quokka then computes a time-variable, spatially uniform heating rate

<script type="math/tex; mode=display">
\Gamma_{\rm PE}(t) = \int_0^t w(t - t') \, \Sigma_{\rm SF}(t') \, dt'
</script>

by convolving the recorded stellar mass increments with a weight function \\(w\\) tabulated against stellar population age. The bundled table `extern/cooling/photoelectric_heating_from_sfh.csv` was generated with SLUG and is normalized so that the solar-neighbourhood star formation rate surface density (2.5 × 10⁻³ M⊙ kpc⁻² yr⁻¹) gives \\(\Gamma\_{\rm PE} = 2 \times 10^{-26}\\) erg s⁻¹ H⁻¹.

This path is only active in 3D. Enabling it requires (all enforced by assertions at startup):

- a cooling table **without** photoelectric heating (a `_noPE` file), since the table would otherwise double-count it;
- `sfh_to_pe_heating_table` pointing at the weight-function CSV;
- either star formation history recording (`sfh_interval` or `sfh_time_interval`) together with `sf_area_kpc2`, or a constant `const_sfr_Msun_per_year_per_kpc2` in place of a real history.

The `TallBoxSf` problem is the reference configuration:

```toml
cooling.hdf5_data_file = "../extern/cooling/CloudyData_UVB=HM2012_shielded_resampled_noPE.h5"
use_sfh_based_pe_heating = true
sfh_to_pe_heating_table = "../extern/cooling/photoelectric_heating_from_sfh.csv"
sfh_time_interval = 3.155760000e+12  # 1e5 yr
sf_area_kpc2 = 1.0  # kpc^2
heating_rate_external = "max(min(1.0, 2.0 - time / (32 * Myr)), 0.0024) * 2e-26"
```

## Using cooling in a problem

Tabulated cooling is the only cooling module currently supported: a problem gets radiative cooling by selecting the `EOSTabulated` backend, and `cooling.cooling_table_type` accepts only `"resampled"`. (`EOSMicrophysics` evolves its own thermochemistry through the reaction network, but that is the chemistry burner rather than this cooling operator, and is limited to Pop III problems.) The rest of this section covers setting up and using the tabulated backend.

### 1. Select the tabulated backend

```cpp
template <> struct quokka::EOS_Traits<MyProblem> {
    static constexpr double gamma = 5. / 3.;
    static constexpr double mean_molecular_weight = C::m_u;
    using EOSBackend = quokka::EOSTabulated<MyProblem>;
};
```

### 2. Point the input file at a table

```toml
cooling.hdf5_data_file = "../extern/cooling/CloudyData_UVB=HM2012_resampled.h5"
```

Quokka reads the table at startup and calls `quokka::ResampledCooling::computeCooling()` as a Strang-split operator each timestep. Cooling requires a non-isothermal EOS; it aborts if `gamma = 1`. It also cannot be combined with photoionization (`photochemistry.enabled = 1`), which models the same hydrogen thermochemistry — see [Photoionization](photoionization.md).

### 3. Choose how photoelectric heating is treated

The table file and `use_sfh_based_pe_heating` together select one of three mutually exclusive treatments of grain photoelectric heating:

| Treatment | Table file | `use_sfh_based_pe_heating` |
|-----------|-----------|----------------------------|
| No photoelectric heating | `_noPE` | `0` (default) |
| Constant PE heating, baked into the table | without `_noPE` | `0` (default) |
| Time-variable PE heating from the star formation rate | `_noPE` | `1` |

- **Ignore photoelectric heating** — use a `_noPE` table. Appropriate where the ISRF is irrelevant or supplied by other means.
- **Constant PE heating** — use a table without `_noPE`. The Wolfire et al. (2003) rate for a fixed \\(G\_0 = 1.7\\) is baked in at table-generation time, so it costs nothing at runtime but cannot respond to the simulation.
- **Star-formation-rate-based PE heating** — use a `_noPE` table *and* set `use_sfh_based_pe_heating = 1`, so the ISRF tracks the stars the simulation forms. See [the section below](#star-formation-history-based-photoelectric-heating) for the required parameters.

The third option requires a `_noPE` table because the SFH-based rate **replaces** the table's static PE term rather than supplementing it; pairing it with a PE-inclusive table would double-count the heating. Quokka enforces this at startup by checking the table's `include_pe` attribute (the filename is only a naming convention) and aborting on a mismatch.

An external `heating_rate_external` expression is independent of this choice and is simply added to whichever rate applies.

### 4. Access thermodynamic quantities in callbacks

Prefer `quokka::EOS<problem_t>`, which forwards to the selected backend and therefore works for any problem:

```cpp
const Real T = quokka::EOS<MyProblem>::ComputeTgasFromEint(rho, Eint);
const Real Eint = quokka::EOS<MyProblem>::ComputeEintFromTgas(rho, T);   // root-finds through the table
const Real K = quokka::EOS<MyProblem>::ComputeEntropyFromRhoEint(rho, Eint);  // EOSTabulated only
```

For table-specific quantities, the GPU-safe helpers in `src/cooling/ResampledCooling.hpp` interpolate directly. They take a `resampledGpuConstTables` handle, obtained inside `QuokkaSimulation` member functions from `resampledTables_.const_tables()` (or `const_tables_host()` for host-side use), and captured by value into the lambda:

```cpp
#include "cooling/ResampledCooling.hpp"

auto tables = resampledTables_.const_tables();
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    // given rho (g/cm³) and Eint (erg/cm³):
    const Real P = quokka::ResampledCooling::ComputePressureFromRhoEint(rho, Eint, tables);
    const Real cs = quokka::ResampledCooling::ComputeSoundSpeedFromRhoEint(rho, Eint, tables);
    const Real Edot = quokka::ResampledCooling::resampled_cooling_function(rho, Eint, tables);
});
```

### 5. Cooling length estimate

```cpp
const Real l_cool = quokka::ResampledCooling::ComputeCoolingLength(rho, Eint, tables);
```

This returns \\(c\_s \, t\_{\rm cool}\\) at the given state and is useful for AMR refinement criteria; `ShockCloud` uses it to tag cells whose cooling length is unresolved.

## Python utilities

The scripts in `extern/cooling/` support table generation and analysis:

| Script | Purpose |
|--------|---------|
| `resample_grackle_cooling_tables.py` | Generate Grackle-based cooling tables from the Cloudy HDF5 data. |
| `resample_grackle_cooling_tables.sh` | Shell wrapper that regenerates all four bundled Grackle tables. |
| `resample_cloudy_cooling_tables.py` | Generate cooling tables directly from Cloudy output (used for ISRF table). |
| `integrate_cooling_zone.py` | Numerically integrate a single cooling zone and compare against Grackle directly. |
| `test_cooling_onezone.py` | One-zone cooling test using `integrate_cooling_zone.py`. |
| `grackle_tables.py` | Low-level reader for the raw Grackle Cloudy HDF5 data, and the reference implementation of the cooling, photoelectric, and Compton terms. |

Install Python dependencies with:

```bash
pip install -r extern/cooling/requirements.txt
```

## Test problem

The `ResampledCoolingTest` problem (`src/problems/ResampledCoolingTest/`) verifies the cooling integrator by comparing the simulated temperature evolution of a single cell against a Grackle reference solution. The test passes when the L1 error between the two trajectories is below 2%.

To build and run:

```bash
quokka buildrun -d 1d ResampledCoolingTest
```
