# BinaryOrbitCIC reports orbit deviation failures without failing the test

Severity: High

## Bug

`src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp` checks `max_deviation` against a tolerance in the normal and restart-refactor branches. When the deviation is too large, both branches print `"Test failed"`, but neither increments `status`.

That means a regression can exceed the accepted orbit-separation error and still return success as long as no particle-count check fails. This masks physics or restart/refinement regressions in CI.

## Patch

```diff
diff --git a/src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp b/src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp
--- a/src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp
+++ b/src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp
@@
 			if (max_deviation < max_err_tol) {
 				amrex::Print() << "Test passed\n";
 			} else {
+				status += 1;
 				amrex::Print() << "Test failed\n";
 			}
@@
 			if (max_deviation < max_err_tol) {
 				amrex::Print() << "Test passed\n";
 			} else {
+				status += 1;
 				amrex::Print() << "Test failed\n";
 			}
```
