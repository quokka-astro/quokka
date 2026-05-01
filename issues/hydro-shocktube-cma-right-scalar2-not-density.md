# HydroShocktubeCMA right boundary writes scalar2 as a fraction instead of partial density

Severity: High

`src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp` stores mass scalars as partial densities. The right boundary correctly multiplies scalar 0 and scalar 1 by `rho_R`, but scalar 2 omits the final multiplication:

```cpp
high_bdr_cells[scalar0_index + 2] = 1 - 0.1 - 0.3 * pow(sin(20 * 3.14 * 1), 2) * rho_R;
```

Because `rho_R = 0.125`, this writes an O(1) mass fraction into a partial-density slot. Ghost cells therefore violate the scalar-density convention and the mass-fraction sum near the right boundary can be far above the gas density.

Patch:

```diff
diff --git a/src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp b/src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp
--- a/src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp
+++ b/src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp
@@
-	high_bdr_cells[HydroSystem<ShocktubeProblem>::scalar0_index + 2] = 1 - 0.1 - 0.3 * pow(sin(20 * 3.14 * 1), 2) * rho_R;
+	high_bdr_cells[HydroSystem<ShocktubeProblem>::scalar0_index + 2] = (1 - 0.1 - 0.3 * pow(sin(20 * 3.14 * 1), 2)) * rho_R;
```
