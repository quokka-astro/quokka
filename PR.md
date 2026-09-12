## Refine max signal speed abort: count-based threshold

Building on PR #1459, this PR refines the signal speed abort mechanism. Instead of aborting immediately when the signal speed exceeds the threshold, the code now counts the number of occurrences and only aborts after `max_signal_counts` consecutive exceedances.

### Changes

- `src/simulation.hpp`:
  - Added `static constexpr int max_signal_counts = 1000` — number of allowed exceedances before abort (compile-time constant, not a runtime parameter)
  - Added `signalSpeedExceedCount_` and `particleSpeedExceedCount_` member variables to track occurrences
  - Modified abort logic: print a `[WARNING]` with current count on each exceedance; abort with `[FATAL]` only when count reaches `max_signal_counts`

### Motivation

A single timestep may occasionally produce a spuriously high signal speed (e.g. at refinement boundaries or during feedback injection). The original immediate-abort behavior was too aggressive. The count-based approach tolerates transient spikes while still catching genuinely runaway simulations.
