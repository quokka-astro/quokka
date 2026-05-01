# ParticleSF validation accepts arbitrarily low star production

Severity: High

`src/problems/ParticleSF/testParticleSF.cpp` checks several expected star-formation statistics using signed relative differences:

```cpp
if (!((m_star_high_tot - exp_m_star_high_total) / exp_m_star_high_total < tol_m_star_high_tot)) { ... }
```

If the simulation underproduces stars, the ratio is negative and therefore always less than the positive tolerance, even when the magnitude of the error is huge. The same pattern is used for high-mass count, low-mass count, and total stellar mass conservation, so important regressions can pass silently.

Patch:

```diff
diff --git a/src/problems/ParticleSF/testParticleSF.cpp b/src/problems/ParticleSF/testParticleSF.cpp
--- a/src/problems/ParticleSF/testParticleSF.cpp
+++ b/src/problems/ParticleSF/testParticleSF.cpp
@@
-			if (!((m_star_high_tot - exp_m_star_high_total) / exp_m_star_high_total < tol_m_star_high_tot)) {
+			if (!(std::abs(m_star_high_tot - exp_m_star_high_total) / exp_m_star_high_total < tol_m_star_high_tot)) {
@@
-			if (!((n_star_high - exp_n_star_high_total) / exp_n_star_high_total < tol_n_star_high)) {
+			if (!(std::abs(n_star_high - exp_n_star_high_total) / exp_n_star_high_total < tol_n_star_high)) {
@@
-			if (!((n_star_low - exp_n_star_low_total) / exp_n_star_low_total < tol_n_star_low)) {
+			if (!(std::abs(n_star_low - exp_n_star_low_total) / exp_n_star_low_total < tol_n_star_low)) {
@@
-			if (!((m_star_tot - m_gas_change) / m_gas_change < tol_m_star_tot)) {
+			if (!(std::abs(m_star_tot - m_gas_change) / m_gas_change < tol_m_star_tot)) {
@@
-		if (!((m_star_tot2 - m_gas_change2) / m_star_tot2 < tol_m_star_tot2)) {
+		if (!(std::abs(m_star_tot2 - m_gas_change2) / m_star_tot2 < tol_m_star_tot2)) {
```
