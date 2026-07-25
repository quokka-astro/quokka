# Optically-Thin Radiative Cooling

Quokka supports optically-thin radiative cooling via pre-computed tables of cooling rate, temperature, sound speed, pressure, and entropy as a function of gas density and specific internal energy. The cooling source term is applied as a Strang-split operator each hydrodynamic timestep.

## Physical model

The cooling source term modifies the gas total energy and internal energy each timestep:

<script type="math/tex; mode=display">
\frac{\partial E}{\partial t}\bigg|_{\rm cool} = \dot{E}_{\rm cool} + \dot{E}_{\rm heat}
</script>

where \\(\dot{E}\_{\rm cool} = \dot{E}\_{\rm cool}(\rho, e\_{\rm int}) \cdot \rho^2\\) is interpolated from a table indexed by density \\(\rho\\) and specific internal energy \\(e\_{\rm int} = E\_{\rm int}/\rho\\), and \\(\dot{E}\_{\rm heat}\\) is an optional constant heating rate per hydrogen atom multiplied by \\(n\_H\\).

Within each cell the ODE

<script type="math/tex; mode=display">
\frac{d E_{\rm int}}{d t} = \dot{E}_{\rm cool} + \dot{E}_{\rm heat}
</script>

is integrated adaptively using an embedded RK2 (Heun's method) sub-cycle. Density is held constant during the integration (isochoric cooling).

## Cooling tables

The tables bundled with Quokka were generated from Grackle's Cloudy data using the scripts in `extern/cooling/`. Each table covers a 2D grid of \\((\rho, e\_{\rm int})\\) with `fast_log` coordinate spacing (a bit-level approximation to log₂) and stores five output quantities as raw physical values. The indices are:

| Index | Quantity | Units |
|-------|----------|-------|
| 0 | Cooling rate \\(\dot{E}\_{\rm cool}/\rho^2\\) | erg cm³ g⁻² s⁻¹ |
| 1 | Temperature | K |
| 2 | Sound speed | cm s⁻¹ |
| 3 | Pressure | dyn cm⁻² |
| 4 | Entropy \\(K = P/\rho^\gamma\\) | erg cm² g⁻⁵/³ |

Available table files:

| File | UV background | Photoelectric heating |
|------|---------------|-----------------------|
| `CloudyData_UVB=HM2012_resampled.h5` | HM2012 | included |
| `CloudyData_UVB=HM2012_resampled_noPE.h5` | HM2012 | excluded |
| `CloudyData_UVB=HM2012_shielded_resampled.h5` | HM2012 (shielded) | included |
| `CloudyData_UVB=HM2012_shielded_resampled_noPE.h5` | HM2012 (shielded) | excluded |
| `isrf_1000Go_grains_resampled.h5` | ISRF 1000 G₀ + grains | included |

To regenerate the Grackle-based tables, run `extern/cooling/resample_grackle_cooling_tables.sh`. To regenerate `isrf_1000Go_grains_resampled.h5`, run `extern/cooling/resample_cloudy_cooling_tables.py` directly.

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
| `cooling.enabled` | bool (0/1) | `0` | Enable optically-thin cooling as a Strang-split source term. |
| `cooling.cooling_table_type` | string | `"resampled"` | Table type. Only `"resampled"` is supported. |
| `cooling.hdf5_data_file` | string | **required** | Path to the cooling table HDF5 file. |
| `cooling.read_tables_even_if_disabled` | bool (0/1) | `0` | Read tables at startup even if cooling is disabled (useful for diagnostics). |
| `heating_rate_external` | string | `""` | AMReX parser expression for a time-variable external heating rate per H atom (erg s⁻¹ H⁻¹). Variables: `time`, `dt`. |

See also [Runtime parameters](parameters.md) for the full parameter table.

## Using cooling in a problem

### 1. Enable cooling in the input file

```toml
cooling.enabled = 1
cooling.hdf5_data_file = "../extern/cooling/CloudyData_UVB=HM2012_resampled.h5"
```

### 2. Enable cooling in the problem traits

```cpp
template <> struct Physics_Traits<MyProblem> {
    // ...
    static constexpr bool is_cooling_enabled = true;
};
```

Quokka will automatically call `quokka::ResampledCooling::computeCooling()` as a Strang-split operator when `is_cooling_enabled = true` and `cooling.enabled = 1`.

### 3. Access thermodynamic quantities in callbacks

The GPU-safe helper functions in `src/cooling/ResampledCooling.hpp` interpolate the table at any \\((\rho, E\_{\rm int})\\) point:

```cpp
#include "cooling/ResampledCooling.hpp"

// inside a GPU lambda, given rho (g/cm³) and Eint (erg/cm³):
const Real T = quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, tables);
const Real P = quokka::ResampledCooling::ComputePressureFromRhoEint(rho, Eint, tables);
const Real cs = quokka::ResampledCooling::ComputeSoundSpeedFromRhoEint(rho, Eint, tables);
const Real K  = quokka::ResampledCooling::ComputeEntropyFromRhoEint(rho, Eint, tables);

// invert T → Eint (uses root finding on the T(Eint) table column)
const Real Eint = quokka::ResampledCooling::ComputeEgasFromTgas(rho, T_target, tables);
```

The `tables` argument is a `resampledGpuConstTables` struct obtained from `resampled_tables::const_tables()`. In problem callbacks that receive `SimulationData<problem_t> &simdata`, obtain it via:

```cpp
auto tables = simdata.coolingTables.const_tables();
```

### 4. Cooling length estimate

```cpp
const Real l_cool = quokka::ResampledCooling::ComputeCoolingLength(rho, Eint, tables);
```

This returns \\(c\_s \, t\_{\rm cool}\\) at the given state and is useful for AMR refinement criteria.

## Python utilities

The scripts in `extern/cooling/` support table generation and analysis:

| Script | Purpose |
|--------|---------|
| `resample_grackle_cooling_tables.py` | Generate Grackle-based cooling tables from the Cloudy HDF5 data. |
| `resample_grackle_cooling_tables.sh` | Shell wrapper that regenerates all four bundled Grackle tables. |
| `resample_cloudy_cooling_tables.py` | Generate cooling tables directly from Cloudy output (used for ISRF table). |
| `integrate_cooling_zone.py` | Numerically integrate a single cooling zone and compare against Grackle directly. |
| `test_cooling_onezone.py` | One-zone cooling test using `integrate_cooling_zone.py`. |
| `grackle_tables.py` | Low-level reader for the raw Grackle Cloudy HDF5 data. |

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

<!-- CI validation touch: remove before merge. -->
