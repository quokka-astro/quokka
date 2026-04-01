## Summary

- Add `src/util/time_units.hpp` with a `queryTime` utility that reads time-valued ParmParse parameters as strings and converts optional physical unit suffixes (`yr`, `kyr`, `Myr`, `Gyr`) to CGS seconds. Conversion uses the Julian year (365.25 days = 3.15576e7 s).
- Replace all time-parameter `pp.query` calls in `src/simulation.hpp` (`stop_time`, `constant_dt`, `initial_dt`, `max_dt`, `dt_cutoff`, `plottime_interval`, `checkpointtime_interval`, `sfh_time_interval`) and `src/io/DiagBase.cpp` (`time_int`) with `queryTime`.
- Update `inputs/ParticleSinkFormation.toml` and `.in` to demonstrate the new syntax (e.g. `quokka.plt.time_int = "1.0_Myr"`).

## Backwards compatibility

Plain CGS values (e.g. `3.15576e13`) continue to work unchanged in both `.toml` and `.in` formats.

## Usage

```toml
# .toml format (string must be quoted)
quokka.plt.time_int = "1.0_Myr"
stop_time = "2.5_Gyr"
initial_dt = "0.1_kyr"

# .in format (unquoted)
quokka.plt.time_int = 1.0_Myr
stop_time = 2.5_Gyr
initial_dt = 0.1_kyr
```

Supported units: `yr`, `kyr`, `Myr`, `Gyr` (case-sensitive).
