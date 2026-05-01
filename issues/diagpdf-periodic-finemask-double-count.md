# DiagPDF double-counts covered coarse cells on periodic AMR domains

Severity: High

## Explanation

`DiagPDF::processDiag` masks coarse cells covered by finer levels before accumulating histogram weights. For non-finest levels it builds the mask with:

```cpp
amrex::makeFineMask(*a_state[lev], *a_state[lev + 1], amrex::IntVect(0), m_refRatio[lev], amrex::Periodicity::NonPeriodic(), 1, 0);
```

This ignores the actual simulation periodicity. On periodic AMR domains, fine grids that cover coarse cells across a periodic boundary are not represented correctly in the mask. Those coarse cells remain active and are accumulated along with the fine cells, so mass-, volume-, and cell-count histograms can double-count covered regions near periodic boundaries.

The projection path already uses `geom[lev].periodicity()` for the same kind of fine-cover mask, which is the correct behavior.

## Patch

Use the stored level geometry periodicity when constructing the fine mask.

```diff
diff --git a/src/io/DiagPDF.H b/src/io/DiagPDF.H
--- a/src/io/DiagPDF.H
+++ b/src/io/DiagPDF.H
@@
-				mask =
-				    amrex::makeFineMask(*a_state[lev], *a_state[lev + 1], amrex::IntVect(0), m_refRatio[lev], amrex::Periodicity::NonPeriodic(), 1, 0);
+				mask = amrex::makeFineMask(*a_state[lev], *a_state[lev + 1], amrex::IntVect(0), m_refRatio[lev],
+							   m_geoms[lev].periodicity(), 1, 0);
 			}
```
