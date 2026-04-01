# Physical Time Units Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow time-valued runtime parameters to use physical unit suffixes (e.g. `1.0_Myr`) instead of raw CGS seconds.

**Architecture:** Add a `queryTime` wrapper in `src/util/time_units.hpp` that reads a ParmParse entry as a string, detects an optional `_<unit>` suffix, and converts to CGS seconds. Replace all time-parameter `pp.query` calls in `simulation.hpp` and `DiagBase.cpp` with `queryTime`. Update the `ParticleSinkFormation` input files to use the new syntax.

**Tech Stack:** C++20, AMReX ParmParse, TOML / `.in` input formats.

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `src/util/time_units.hpp` | `queryTime` utility + time unit constants |
| Modify | `src/simulation.hpp` | Replace 8 `pp.query` calls with `queryTime` |
| Modify | `src/io/DiagBase.cpp` | Replace 1 `pp.query` call with `queryTime` |
| Modify | `inputs/ParticleSinkFormation.toml` | Use `"1.0_Myr"` / `"0.1_Myr"` syntax |
| Modify | `inputs/ParticleSinkFormation.in` | Use `1.0_Myr` / `0.1_Myr` syntax (unquoted) |

---

## Task 1: Create `src/util/time_units.hpp`

**Files:**
- Create: `src/util/time_units.hpp`

- [ ] **Step 1: Create the header**

```cpp
#ifndef TIME_UNITS_HPP_
#define TIME_UNITS_HPP_
/// \file time_units.hpp
/// \brief Utility for parsing time-valued ParmParse entries with optional physical unit suffixes.
/// Supported suffixes: _yr, _kyr, _Myr, _Gyr (case-sensitive).
/// Examples: "1.0_Myr", "500_kyr", "1.3_Gyr", or plain "3.15576e13" (CGS seconds).

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

namespace quokka
{

/// Time unit conversion factors to CGS seconds (Julian year = 365.25 days).
inline constexpr double yr_in_s = 3.15576e7;
inline constexpr double kyr_in_s = 3.15576e10;
inline constexpr double Myr_in_s = 3.15576e13;
inline constexpr double Gyr_in_s = 3.15576e16;

/// \brief Parse a time string that may carry a physical unit suffix.
///
/// Accepted formats:
///   "3.15576e13"      — plain CGS seconds, returned as-is
///   "1.0_Myr"         — converted to CGS seconds
///   "500_kyr"         — converted to CGS seconds
///
/// Supported units: yr, kyr, Myr, Gyr (case-sensitive).
/// Aborts with a descriptive message on unrecognised unit.
///
/// \param s    the string to parse
/// \param name parameter name used in error messages
/// \return value in CGS seconds
inline auto parseTimeString(const std::string &s, const std::string &name) -> amrex::Real
{
	static const std::unordered_map<std::string, double> unitMap = {
	    {"yr", yr_in_s}, {"kyr", kyr_in_s}, {"Myr", Myr_in_s}, {"Gyr", Gyr_in_s}};

	const auto pos = s.rfind('_');
	if (pos != std::string::npos) {
		const std::string numStr = s.substr(0, pos);
		const std::string unit = s.substr(pos + 1);
		auto const it = unitMap.find(unit);
		if (it == unitMap.end()) {
			amrex::Abort("queryTime: unrecognised time unit '" + unit + "' for parameter '" + name +
				     "'. Supported units: yr, kyr, Myr, Gyr.");
		}
		return static_cast<amrex::Real>(std::stod(numStr) * it->second);
	}
	return static_cast<amrex::Real>(std::stod(s));
}

/// \brief Drop-in replacement for pp.query() for time-valued parameters.
///
/// Reads the parameter as a string (works for both .in and .toml formats),
/// then calls parseTimeString. If the parameter is absent, \p val is unchanged.
///
/// \param pp   ParmParse instance (any prefix)
/// \param name parameter name
/// \param val  output value in CGS seconds; unchanged if parameter not found
/// \return true if the parameter was found, false otherwise
inline auto queryTime(const amrex::ParmParse &pp, const std::string &name, amrex::Real &val) -> bool
{
	std::string str;
	if (pp.query(name.c_str(), str) == 0) {
		return false;
	}
	val = parseTimeString(str, name);
	return true;
}

} // namespace quokka

#endif // TIME_UNITS_HPP_
```

- [ ] **Step 2: Commit**

```bash
git add src/util/time_units.hpp
git commit -m "feat: add queryTime utility for physical time unit suffixes"
```

---

## Task 2: Update `src/io/DiagBase.cpp`

**Files:**
- Modify: `src/io/DiagBase.cpp`

- [ ] **Step 1: Add the include**

At the top of `src/io/DiagBase.cpp`, add after the existing includes:

```cpp
#include "util/time_units.hpp"
```

- [ ] **Step 2: Replace the `time_int` query**

Find (line ~11):
```cpp
	pp.query("time_int", m_time_interval); // time_int takes precedence over per
```
Replace with:
```cpp
	quokka::queryTime(pp, "time_int", m_time_interval); // time_int takes precedence over per; supports unit suffixes (e.g. "1.0_Myr")
```

- [ ] **Step 3: Commit**

```bash
git add src/io/DiagBase.cpp
git commit -m "feat: use queryTime for DiagBase time_int parameter"
```

---

## Task 3: Update `src/simulation.hpp`

**Files:**
- Modify: `src/simulation.hpp`

- [ ] **Step 1: Add the include**

In `src/simulation.hpp`, find the block of utility includes (near top, after AMReX includes). Add:
```cpp
#include "util/time_units.hpp"
```

- [ ] **Step 2: Replace the 8 time-parameter queries in `readParameters()`**

Replace each of the following (exact lines shown below):

```cpp
	pp.query("constant_dt", constantDt_);
	pp.query("initial_dt", initDt_);
	pp.query("max_dt", maxDt_);
```
with:
```cpp
	quokka::queryTime(pp, "constant_dt", constantDt_);
	quokka::queryTime(pp, "initial_dt", initDt_);
	quokka::queryTime(pp, "max_dt", maxDt_);
```

Replace:
```cpp
	pp.query("stop_time", stopTime_);
```
with:
```cpp
	quokka::queryTime(pp, "stop_time", stopTime_);
```

Replace:
```cpp
	pp.query("dt_cutoff", dtCutoff_);
```
with:
```cpp
	quokka::queryTime(pp, "dt_cutoff", dtCutoff_);
```

Replace:
```cpp
	pp.query("plottime_interval", plotTimeInterval_);
```
with:
```cpp
	quokka::queryTime(pp, "plottime_interval", plotTimeInterval_);
```

Replace:
```cpp
	pp.query("checkpointtime_interval", checkpointTimeInterval_);
```
with:
```cpp
	quokka::queryTime(pp, "checkpointtime_interval", checkpointTimeInterval_);
```

Replace:
```cpp
	pp.query("sfh_time_interval", sfh_time_interval_);
```
with:
```cpp
	quokka::queryTime(pp, "sfh_time_interval", sfh_time_interval_);
```

- [ ] **Step 3: Commit**

```bash
git add src/simulation.hpp
git commit -m "feat: use queryTime for all time-valued parameters in simulation.hpp"
```

---

## Task 4: Build and verify compilation

**Files:** (none new)

- [ ] **Step 1: Source environment and build**

```bash
source ~/.local/bin/quokka.rc
quokka build 3d ParticleSinkFormation --root "$REPO_ROOT"
```

Expected: build completes with no errors. Warnings about unused variables are acceptable; errors are not.

- [ ] **Step 2: Fix any compile errors**

If the build fails due to missing include path, check that `src/util/time_units.hpp` is reachable from the include directories used by `DiagBase.cpp` and `simulation.hpp`. The path `"util/time_units.hpp"` is relative to `src/`, which is the root include directory for the project.

---

## Task 5: Update `inputs/ParticleSinkFormation.toml`

**Files:**
- Modify: `inputs/ParticleSinkFormation.toml`

Current values and their replacements (all equivalent in CGS):

| Parameter | Current value | New value |
|-----------|--------------|-----------|
| `initial_dt` | `3.15576e12` | `"0.1_Myr"` |
| `quokka.plt.time_int` | `3.155760000e+13` | `"1.0_Myr"` |
| `quokka.slice_x.time_int` | `3.155760000e+13` | `"1.0_Myr"` |
| `quokka.part.time_int` | `3.155760000e+13` | `"1.0_Myr"` |

- [ ] **Step 1: Apply the changes**

In `inputs/ParticleSinkFormation.toml`:

Replace `initial_dt = 3.15576e12` with `initial_dt = "0.1_Myr"`.

Replace `quokka.plt.time_int = 3.155760000e+13 # Output cadence (in time intervals). 1 Myr, expected outputs are: 8 12 17`
with `quokka.plt.time_int = "1.0_Myr" # Output cadence. 1 Myr, expected outputs are: 8 12 17`

Replace `quokka.slice_x.time_int = 3.155760000e+13 # Output cadence (in time intervals). 1 Myr, expected outputs are: 8 12 17`
with `quokka.slice_x.time_int = "1.0_Myr" # Output cadence. 1 Myr, expected outputs are: 8 12 17`

Replace `quokka.part.time_int = 3.155760000e+13 # Output cadence (in time intervals). 1 Myr, expected outputs are: 8 12 17`
with `quokka.part.time_int = "1.0_Myr" # Output cadence. 1 Myr, expected outputs are: 8 12 17`

- [ ] **Step 2: Commit**

```bash
git add inputs/ParticleSinkFormation.toml
git commit -m "test: update ParticleSinkFormation.toml to use physical time unit syntax"
```

---

## Task 6: Update `inputs/ParticleSinkFormation.in`

**Files:**
- Modify: `inputs/ParticleSinkFormation.in`

Same substitutions as Task 5 but **without quotes** (`.in` format uses unquoted tokens).

- [ ] **Step 1: Apply the changes**

In `inputs/ParticleSinkFormation.in`:

Replace `initial_dt = 3.15576e12` with `initial_dt = 0.1_Myr`.

Replace `quokka.plt.time_int  = 2e13` with `quokka.plt.time_int = 1.0_Myr`.

Note: `2e13` ≠ `3.15576e13` — the `.in` file uses `2e13` (a rounded value). Use `1.0_Myr` (= 3.15576e13) to match the `.toml` file.

Replace `quokka.slice_x.time_int    = 4e13` with `quokka.slice_x.time_int = 1.0_Myr`.

Replace `quokka.part.time_int = 4e13` with `quokka.part.time_int = 1.0_Myr`.

- [ ] **Step 2: Commit**

```bash
git add inputs/ParticleSinkFormation.in
git commit -m "test: update ParticleSinkFormation.in to use physical time unit syntax"
```

---

## Task 7: Run the test and validate output

**Files:** (none)

- [ ] **Step 1: Run the test**

```bash
quokka run 3d ParticleSinkFormation --root "$REPO_ROOT"
```

- [ ] **Step 2: Compare output to reference**

Expected plotfile steps based on `tests/00ref`: check that the same step numbers appear in `$REPO_ROOT/tests/`.

```bash
ls "$REPO_ROOT/tests/" | grep '^plt'
ls "$REPO_ROOT/tests/00ref/" | grep '^plt'
```

Both listings should show the same step numbers (e.g. `plt0000008`, `plt0000012`, `plt0000017`).

- [ ] **Step 3: Diff Headers**

```bash
diff "$REPO_ROOT/tests/00ref/plt0000008/Header" "$REPO_ROOT/tests/plt0000008/Header"
```

Expected: no differences (or only the timestamp line if present).

- [ ] **Step 4: Commit (if not already clean)**

No new files to commit at this stage. If any fixups were needed, commit them.

---

## Task 8: Write PR.md

**Files:**
- Create: `PR.md`

- [ ] **Step 1: Create PR.md**

```markdown
## Summary

- Add `src/util/time_units.hpp` with a `queryTime` utility that parses time-valued ParmParse parameters with optional physical unit suffixes (`yr`, `kyr`, `Myr`, `Gyr`), converting to CGS seconds.
- Replace all time-parameter `pp.query` calls in `src/simulation.hpp` and `src/io/DiagBase.cpp` with `queryTime`, enabling syntax like `stop_time = "1.0_Gyr"` or `quokka.plt.time_int = "10_Myr"`.
- Update `inputs/ParticleSinkFormation.toml` and `.in` to demonstrate the new syntax.

## Backwards compatibility

Plain CGS values (e.g. `3.15576e13`) continue to work unchanged.
```

- [ ] **Step 2: Commit**

```bash
git add PR.md
git commit -m "docs: add PR description"
```
