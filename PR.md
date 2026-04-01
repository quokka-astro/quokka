### Description
Add support for physical time unit expressions in runtime parameter files by leveraging AMReX's built-in `queryWithParser` mechanism.

Time-valued parameters (`stop_time`, `initial_dt`, `max_dt`, `dt_cutoff`, `plottime_interval`, `checkpointtime_interval`, `sfh_time_interval`, and diagnostic `time_int`) now accept math expressions referencing the named constants `yr`, `kyr`, `Myr`, and `Gyr` (Julian year = 365.25 days = 3.15576×10⁷ s). Full arithmetic expressions are supported: `stop_time = "2.5*Myr + 500*kyr"`.

Implementation: `quokka::registerTimeUnitConstants()` (new, in `src/util/time_units.hpp`) adds yr/kyr/Myr/Gyr to ParmParse under the `quokka_time_units` prefix and calls `ParmParse::SetParserPrefix("quokka_time_units")` once at the start of `readParameters()`. All time-parameter reads use `pp.queryWithParser()`. Plain CGS values (e.g. `3.15576e13`) continue to work unchanged.

**Usage:**
```toml
# .toml format (expression must be quoted)
quokka.plt.time_int = "1.0*Myr"
stop_time = "2.5*Gyr"
initial_dt = "0.1*kyr"
# arithmetic expressions also work:
# stop_time = "2.0*Myr + 500*kyr"
```
```
# .in format (unquoted)
quokka.plt.time_int = 1.0*Myr
stop_time = 2.5*Gyr
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
