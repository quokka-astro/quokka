### Description

Add support for physical time unit suffixes (`yr`, `kyr`, `Myr`, `Gyr`) in runtime parameter files, so users can write e.g. `quokka.plt.time_int = "1.0_Myr"` instead of `quokka.plt.time_int = 3.155760000e+13`. Conversion uses the Julian year (1 year = 365.25 days = 3.15576e7 s).

A new utility `quokka::queryTime()` in `src/util/time_units.hpp` replaces `pp.query()` for all time-valued parameters in `src/simulation.hpp` and `src/io/DiagBase.cpp`. Plain CGS values continue to work unchanged (backwards compatible).

**Supported parameters:** `stop_time`, `constant_dt`, `initial_dt`, `max_dt`, `dt_cutoff`, `plottime_interval`, `checkpointtime_interval`, `sfh_time_interval`, `time_int` (diagnostics).

**Usage:**
```toml
# .toml format (string must be quoted)
quokka.plt.time_int = "1.0_Myr"
stop_time = "2.5_Gyr"
initial_dt = "0.1_kyr"
```
```
# .in format (unquoted)
quokka.plt.time_int = 1.0_Myr
stop_time = 2.5_Gyr
```

### Related issues

N/A

### Checklist

_Before this pull request can be reviewed, all of these tasks should be completed. Denote completed tasks with an `x` inside the square brackets `[ ]` in the Markdown source below:_
- [x] I have added a description (see above).
- [x] I have added a link to any related issues (if applicable; see above).
- [x] I have read the [Contributing Guide](https://github.com/quokka-astro/quokka/blob/development/CONTRIBUTING.md).
- [x] I have added tests for any new physics that this PR adds to the code.
- [ ] *(For quokka-astro org members)* I have manually triggered the GPU tests with the magic comment `/azp run`.
