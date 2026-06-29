# DataTable HDF5 Format + ResampledCooling Redesign

**Branch:** `chong/cooling/use-DataTable2`  
**Date:** 2026-06-29  
**Reference:** ADR `ADR-use-HDF5-datatable-for-cooling.md`, aborted PR #1533

---

## Goal

Update `quokka::DataTable` to read a new, self-describing HDF5 group format, and rewrite `ResampledCooling` to load all five cooling outputs into a single `DataTable<2, 5>` using that reader.

---

## HDF5 File Format

Each HDF5 file may contain one or more named groups (`tab1`, `tab2`, …). Each group is self-contained and follows this layout:

```
/tab1/
  attrs:
    Ndim            int           number of input dimensions
    Nx              int[Ndim]     grid points per input dimension
    Nout            int           number of output quantities
    input_names     str[Ndim]     e.g. ['rho', 'eint']
    output_names    str[Nout]     e.g. ['cooling_rate', 'temperature', ...]
    input_units     str[Ndim]     e.g. ['g/cm^3', 'erg/g']
    output_units    str[Nout]     e.g. ['erg/cm^3/s/(g/cm^3)^2', 'K', ...]
    xlo             float[Ndim]   lower bounds in physical units (not log)
    xhi             float[Ndim]   upper bounds in physical units (not log)
    spacing         str[Ndim]     per-dimension: 'linear', 'log', or 'fast_log'
    [extra attrs]                 domain-specific (e.g. include_pe, cloudy_H_mass_fraction)
  /data             dataset       shape [Nout, Nx[0], Nx[1], ...], row-major, physical values
  /grids/           group (optional, for irregular grids — not yet supported in C++)
    rho             dataset       physical-unit grid points for dim 0
    eint            dataset       physical-unit grid points for dim 1
```

**Resampled cooling file (`tab1`) attributes:**

| Attribute | Value |
|---|---|
| `Ndim` | 2 |
| `Nx` | `[n_rho, n_eint]` |
| `Nout` | 5 |
| `input_names` | `['rho', 'eint']` |
| `output_names` | `['cooling_rate', 'temperature', 'sound_speed', 'pressure', 'entropy']` |
| `input_units` | `['g/cm^3', 'erg/g']` |
| `output_units` | `['erg/cm^3/s/(g/cm^3)^2', 'K', 'cm/s', 'dyne/cm^2', 'erg*cm^2']` |
| `xlo` | `[rho_min, eint_min]` (physical) |
| `xhi` | `[rho_max, eint_max]` (physical) |
| `spacing` | `['fast_log', 'fast_log']` |
| `include_pe` | `0` or `1` |
| `cloudy_H_mass_fraction` | float |

`tab1/data` has shape `[5, n_rho, n_eint]`.

---

## `DataTable` Changes (`src/util/DataTable.hpp`)

### 1. Store physical bounds

`initializeStorage` saves the raw `x_mins` / `x_maxs` into new private members before overwriting `coord_min_` / `coord_max_` with their log-transformed equivalents:

```cpp
std::array<amrex::Real, Ndim> xlo_physical_{};
std::array<amrex::Real, Ndim> xhi_physical_{};
```

Two new public accessors:

```cpp
[[nodiscard]] auto coord_xlo() const -> std::array<amrex::Real, Ndim>;
[[nodiscard]] auto coord_xhi() const -> std::array<amrex::Real, Ndim>;
```

### 2. New `H5Reader` (replaces old one)

Old signature (removed):
```cpp
static auto H5Reader(const std::string &file_path,
                     const std::string &dataset_path,
                     const std::vector<std::string> &coord_names,
                     int is_fast_log = 0,
                     std::array<std::pair<amrex::Real, amrex::Real>, Ndim> *coord_bounds = nullptr,
                     bool *include_pe = nullptr) -> DataTable;
```

New signature:
```cpp
static auto H5Reader(const std::string &file_path,
                     const std::string &group_name = "tab1") -> DataTable;
```

Behaviour:
- Opens `/<group_name>` in the HDF5 file.
- Reads all standard attributes (`Ndim`, `Nx`, `Nout`, `xlo`, `xhi`, `spacing`, `input_names`, `output_names`, `input_units`, `output_units`).
- Asserts `Ndim == template Ndim` and `Nout == template Nout`.
- Reads the `data` dataset (flat, row-major) into a buffer and calls `initializeCommonFlat`.
- Sets all metadata via `setMetadata` (made public or called internally).
- If a `grids` subgroup exists, calls `amrex::Abort("DataTable currently does not support irregular grids")`.
- All data is broadcast across MPI ranks (same pattern as current code).

### 3. `setMetadata` visibility

Move `setMetadata` from `private` to `public` (or call it internally inside `H5Reader` — either works).

---

## `ResampledCooling` Changes

### `ResampledCooling.hpp`

**Named output index constants:**

```cpp
constexpr int COOLING_RATE_IDX = 0;
constexpr int TEMPERATURE_IDX  = 1;
constexpr int SOUND_SPEED_IDX  = 2;
constexpr int PRESSURE_IDX     = 3;
constexpr int ENTROPY_IDX      = 4;
```

**`resampledGpuConstTables` (GPU-side):**

```cpp
struct resampledGpuConstTables {
    quokka::DataTableGpuConst<2, 5> all_tables;
    amrex::Real cloudy_H_mass_fraction;
    amrex::Real eint_min;   // physical erg/g
    amrex::Real eint_max;   // physical erg/g
};
```

**`resampled_tables` (host-side):**

```cpp
class resampled_tables {
  public:
    quokka::DataTable<2, 5> all_tables;
    amrex::Real cloudy_H_mass_fraction;
    bool include_pe;

    [[nodiscard]] auto const_tables() const -> resampledGpuConstTables;
    [[nodiscard]] auto const_tables_host() const -> resampledGpuConstTables;
};
```

**Interpolation call-sites** — the `fast_log` transform is now internal to `DataTable`, so callers pass physical values:

```cpp
// Before:
std::array<amrex::Real, 2> const point = {FastMath::fastlg(rho), FastMath::fastlg(eint)};
const Real val = tables.cooling_rates.interpolate_single(point);

// After:
std::array<amrex::Real, 2> const point = {rho, eint};
const Real val = tables.all_tables.interpolate_single(point, COOLING_RATE_IDX);
```

`ComputeEgasFromTgas` is unchanged: it still reads `tables.eint_min` / `tables.eint_max` for root-finding bounds.

### `ResampledCooling.cpp`

**`readResampledData`:**

```cpp
resampledTables.all_tables = quokka::DataTable<2, 5>::H5Reader(hdf5_file, "tab1");
// include_pe and cloudy_H_mass_fraction read directly from tab1 attrs via HDF5 API
```

**`resampled_tables::const_tables()`:**

```cpp
return resampledGpuConstTables{
    all_tables.const_tables(),
    cloudy_H_mass_fraction,
    all_tables.coord_xlo()[1],  // eint_min — physical erg/g
    all_tables.coord_xhi()[1],  // eint_max — physical erg/g
};
```

---

## Python Script Changes (`extern/cooling/resample_grackle_cooling_tables.py`)

Replace the file-save block. Instead of separate `/metadata`, `/grids`, `/data` groups, write a single `tab1` group with the attributes and a single `[5, n_rho, n_eint]` dataset. The `grids` subgroup inside `tab1` is written with physical-unit coordinate arrays (informational only).

---

## Existing HDF5 Files

All five `.h5` files in `extern/cooling/` must be regenerated with the updated Python script and committed as updated binaries. This step runs locally (requires `grackle`).

---

## Verification

```bash
quokka buildrun -d 1d ResampledCoolingTest
```

The test does a 1-zone cooling integration and compares against `grackle_reference_solution.csv`. Passing this test is the correctness criterion.
