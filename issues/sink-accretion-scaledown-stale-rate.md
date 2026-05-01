# Sink accretion scale-down uses stale accretion rate after clamping

Severity: High

## Explanation

`SinkAccretionUtils::ComputeScaleDown` first clamps very negative relative accretion rates to `-0.25` to avoid removing more than 25% of a cell's mass. It then applies a Jeans-density correction, but that second check still uses the original `accretion_rate_cell` value captured before the clamp.

For example, if the raw rate is `-0.8`, the code stores `-0.25` in the MultiFab but still tests `(1.0 - 0.8) * rho > rho_J`. If the clamped end state `0.75*rho` is still above `rho_J` while the raw end state `0.2*rho` is not, the Jeans correction is skipped. The result is under-accretion and gas left above the intended Jeans threshold around sinks.

The same stale value is also used as the denominator when updating `scale_down`, so any subsequent correction should be based on the current clamped rate.

## Patch

```diff
diff --git a/src/particles/particle_accretion.hpp b/src/particles/particle_accretion.hpp
--- a/src/particles/particle_accretion.hpp
+++ b/src/particles/particle_accretion.hpp
@@
-		const double accretion_rate_cell = local_accretion_rate_arr[bx](i, j, k);
+		double accretion_rate_cell = local_accretion_rate_arr[bx](i, j, k);
 		const double accretion_rate_floor = -0.25;
 		if (accretion_rate_cell < accretion_rate_floor) {
 			// scale down the accretion rate to the minimum allowed value
 			local_accretion_rate_arr[bx](i, j, k) = accretion_rate_floor;
 			local_scale_down_arr[bx](i, j, k) = accretion_rate_floor / accretion_rate_cell;
+			accretion_rate_cell = accretion_rate_floor;
 		}
@@
 			if ((1.0 + accretion_rate_cell) * rho_cell > rho_J) {
 				const double accretion_rate_cell_new = rho_J / rho_cell - 1.0;
 				local_accretion_rate_arr[bx](i, j, k) = accretion_rate_cell_new;
 				local_scale_down_arr[bx](i, j, k) = accretion_rate_cell_new / accretion_rate_cell;
+				accretion_rate_cell = accretion_rate_cell_new;
 			}
```
