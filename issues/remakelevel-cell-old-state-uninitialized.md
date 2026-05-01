# RemakeLevel leaves the cell-centered old state uninitialized

Severity: High

## Bug

`AMRSimulation::RemakeLevel` allocates both `int_state_new_cc` and `int_state_old_cc`, but only fills `int_state_new_cc` before swapping both into the level:

```cpp
amrex::MultiFab int_state_new_cc(ba, dm, ncomp_cc, nghost_cc);
amrex::MultiFab int_state_old_cc(ba, dm, ncomp_cc, nghost_cc);
FillPatch(level, time, int_state_new_cc, ...);
std::swap(int_state_new_cc, state_new_cc_[level]);
std::swap(int_state_old_cc, state_old_cc_[level]);
```

After a regrid/remake, `state_old_cc_[level]` contains uninitialized FAB data. Later AMR fill operations can request both old and new data for time interpolation whenever the requested time is not exactly `tNew_[level]` or `tOld_[level]`. This can inject garbage or NaNs into filled ghost cells and then into evolved states. The face-centered branch does not have the same omission; it fills both temporary old and new arrays before swapping.

## Patch

```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
 	amrex::MultiFab int_state_new_cc(ba, dm, ncomp_cc, nghost_cc);
 	amrex::MultiFab int_state_old_cc(ba, dm, ncomp_cc, nghost_cc);
 	FillPatch(level, time, int_state_new_cc, 0, ncomp_cc, quokka::centering::cc, quokka::direction::na, FillPatchType::fillpatch_function);
+	amrex::MultiFab::Copy(int_state_old_cc, int_state_new_cc, 0, 0, ncomp_cc, nghost_cc);
 	std::swap(int_state_new_cc, state_new_cc_[level]);
 	std::swap(int_state_old_cc, state_old_cc_[level]);
```
