# AMRSimulation::RemakeLevel(...): `int_state_old_cc` is allocated but never filled before `std::swap(int_state_old_cc, state_old_cc_[level])` (`src/simulation.hpp:2386-2389`), so `state_old_cc_[level]` becomes uninitialized after remaking a level

## Summary
`int_state_old_cc` is allocated but never filled before `std::swap(int_state_old_cc, state_old_cc_[level])` (`src/simulation.hpp:2386-2389`), so `state_old_cc_[level]` becomes uninitialized after remaking a level.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::RemakeLevel(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:241`
- Finding tags: correctness

## Proposed Patch
- Initialize or copy-populate the temporary state before swapping it into the live state vector; add a regression test that exercises the remake/remap path.

## Why This Is a Bug
`RemakeLevel(...)` allocates `int_state_old_cc` and then swaps it into `state_old_cc_[level]` without ever populating it. That replaces valid old-state data with uninitialized `MultiFab` contents after regridding/remaking a level, which can break fillpatch interpolation and time interpolation logic that relies on `state_old_cc_`.

## Complete Code Patch
```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
 	amrex::MultiFab int_state_new_cc(ba, dm, ncomp_cc, nghost_cc);
 	amrex::MultiFab int_state_old_cc(ba, dm, ncomp_cc, nghost_cc);
 	FillPatch(level, time, int_state_new_cc, 0, ncomp_cc, quokka::centering::cc, quokka::direction::na, FillPatchType::fillpatch_function);
+	FillPatch(level, time, int_state_old_cc, 0, ncomp_cc, quokka::centering::cc, quokka::direction::na, FillPatchType::fillpatch_function);
 	std::swap(int_state_new_cc, state_new_cc_[level]);
 	std::swap(int_state_old_cc, state_old_cc_[level]);
```
