# FieldLoop total energy omits z-momentum kinetic energy

Severity: High

`src/problems/FieldLoop/testFieldLoop.cpp` initializes `vz = 1.0` and writes `x3Momentum = rho0 * vz`, but the total gas energy only includes `vx` and `vy`:

```cpp
const double Ekin = 0.5 * rho0 * (vx * vx + vy * vy);
```

The conserved state is therefore internally inconsistent from the first step. Any later pressure/internal-energy recovery from total energy subtracts the full kinetic energy, including `x3Momentum`, even though that z kinetic energy was never added. This lowers the recovered thermal pressure by `0.5 * rho0 * vz^2` and can make the field-loop problem evolve with the wrong thermodynamic state.

Patch:

```diff
diff --git a/src/problems/FieldLoop/testFieldLoop.cpp b/src/problems/FieldLoop/testFieldLoop.cpp
--- a/src/problems/FieldLoop/testFieldLoop.cpp
+++ b/src/problems/FieldLoop/testFieldLoop.cpp
@@
-		const double Ekin = 0.5 * rho0 * (vx * vx + vy * vy);
+		const double Ekin = 0.5 * rho0 * (vx * vx + vy * vy + vz * vz);
```
