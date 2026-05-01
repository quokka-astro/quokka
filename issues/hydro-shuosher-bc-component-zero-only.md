# HydroShuOsher only initializes x-boundaries for component 0

Severity: High

`src/problems/HydroShuOsher/testHydroShuOsher.cpp` allocates a `BCRec` for every conserved component, but the x-direction boundary loop writes to `BCs_cc[0]` instead of `BCs_cc[n]`:

```cpp
for (int n = 0; n < ncomp_cc; ++n) {
	BCs_cc[0].setLo(0, amrex::BCType::foextrap);
	BCs_cc[0].setHi(0, amrex::BCType::ext_dir);
	...
}
```

Only component 0 receives the intended x-boundaries. Momentum, energy, and internal energy keep default/uninitialized x-boundary records, so the custom boundary condition machinery can be bypassed or applied inconsistently for those fields.

Patch:

```diff
diff --git a/src/problems/HydroShuOsher/testHydroShuOsher.cpp b/src/problems/HydroShuOsher/testHydroShuOsher.cpp
--- a/src/problems/HydroShuOsher/testHydroShuOsher.cpp
+++ b/src/problems/HydroShuOsher/testHydroShuOsher.cpp
@@
 	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
 	for (int n = 0; n < ncomp_cc; ++n) {
-		BCs_cc[0].setLo(0, amrex::BCType::foextrap); // Dirichlet
-		BCs_cc[0].setHi(0, amrex::BCType::ext_dir);
+		BCs_cc[n].setLo(0, amrex::BCType::foextrap); // Dirichlet
+		BCs_cc[n].setHi(0, amrex::BCType::ext_dir);
```
