# Physical Time Units in Runtime Parameter Files

**Date:** 2026-04-01
**Branch:** chong/io/allow-time-unit

## Summary

Add support for physical time unit suffixes (yr, kyr, Myr, Gyr) in runtime parameter files, so users can write `quokka.plt.time_int = "1.0_Myr"` instead of `quokka.plt.time_int = 3.155760000e+13`.

## Motivation

Time parameters in Quokka input files are specified in CGS seconds, which requires the user to manually convert (e.g., 1 Myr = 3.15576e13 s). Unit suffixes make inputs more readable and less error-prone.

## Design

### Approach: `queryTime` wrapper (Approach B)

Create a single utility function `queryTime(pp, name, val)` as a drop-in replacement for `pp.query(name, val)` for time-valued parameters. This function:

1. Always reads the parameter as a `std::string` (works for both `.in` and `.toml` formats)
2. If the string contains `_<unit>`, splits on `_`, parses the numeric part, and multiplies by the unit conversion factor
3. Otherwise, converts the string directly to `double` via `std::stod`
4. If the parameter is absent, leaves `val` unchanged (same as `pp.query`)

### New file: `src/util/time_units.hpp`

```cpp
// Time unit constants (CGS seconds)
constexpr double yr_in_s  = 3.15576e7;
constexpr double kyr_in_s = 3.15576e10;
constexpr double Myr_in_s = 3.15576e13;
constexpr double Gyr_in_s = 3.15576e16;

// Supported suffixes: yr, kyr, Myr, Gyr
// Falls back to plain double if no suffix present
auto queryTime(amrex::ParmParse const &pp, std::string const &name, amrex::Real &val) -> bool;
```

Error behavior: if the suffix is unrecognized (e.g. `1.0_parsec`), abort with a clear message listing supported units.

### Call sites updated

**`src/simulation.hpp` (`readParameters()`):**
- `stop_time`
- `constant_dt`
- `initial_dt`
- `max_dt`
- `dt_cutoff`
- `plottime_interval`
- `checkpointtime_interval`
- `sfh_time_interval`

**`src/io/DiagBase.cpp`:**
- `time_int`

**Not changed** (not simulation time):
- `max_walltime` (wallclock time)
- Step-count intervals: `plotfile_interval`, `checkpoint_interval`, `statistics_interval`, etc.
- Problem-specific `pp.query` calls in `src/problems/`

### Input file syntax

**`.toml` files** (strings must be quoted):
```toml
quokka.plt.time_int = "1.0_Myr"
initial_dt = "0.1_Myr"
```
Plain numbers still work without quotes: `quokka.plt.time_int = 3.155760000e+13`

**`.in` files** (unquoted tokens):
```
quokka.plt.time_int = 1.0_Myr
initial_dt = 0.1_Myr
```

### Input files updated

- `inputs/ParticleSinkFormation.toml`: `time_int` for plt, slice_x, part changed to `"1.0_Myr"`; `initial_dt` changed to `"0.1_Myr"`
- `inputs/ParticleSinkFormation.in`: same, without quotes

## Validation

Build and run `ParticleSinkFormation` (3D preset). Output in `tests/` must be bit-for-bit identical to `tests/00ref` since the CGS values are unchanged.

## Files Changed

| File | Change |
|------|--------|
| `src/util/time_units.hpp` | New — `queryTime` utility + unit constants |
| `src/simulation.hpp` | Replace 8 `pp.query` calls with `queryTime` |
| `src/io/DiagBase.cpp` | Replace 1 `pp.query` call with `queryTime` |
| `inputs/ParticleSinkFormation.toml` | Use `"1.0_Myr"` syntax |
| `inputs/ParticleSinkFormation.in` | Use `1.0_Myr` syntax |
