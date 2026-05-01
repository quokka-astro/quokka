# SN Galilean-invariance validation is skipped without Python plotting

Severity: High

`src/problems/SN/testSN.cpp` computes the base/boosted temperature and velocity profiles, relative error norms, tolerance checks, and `status = 1` assignment inside `#ifdef HAVE_PYTHON`. When the code is built without Python/matplotlib support, the plotting code is correctly omitted, but so is the numerical validation. In that configuration, the test can return success even if the Galilean-invariance comparison fails completely.

The plotting dependency should not guard correctness checks. Only the `matplotlibcpp` calls and plot-specific vectors need to stay under `#ifdef HAVE_PYTHON`; the extraction loop, norm calculation, and status update should always run on the IO processor.

Patch:

```diff
diff --git a/src/problems/SN/testSN.cpp b/src/problems/SN/testSN.cpp
--- a/src/problems/SN/testSN.cpp
+++ b/src/problems/SN/testSN.cpp
@@
 	// plot the temperature and vx profile along the x axis at the center
 	if (amrex::ParallelDescriptor::IOProcessor()) {
-#ifdef HAVE_PYTHON
 		for (int i = 0; i < nx; ++i) {
 			const double rho = values.at(HydroSystem<SNProblem>::density_index)[i];
 			const double Eint = values.at(HydroSystem<SNProblem>::internalEnergy_index)[i];
@@
 			T[i] = Eint / (rho * CV); // simplified, but good enough for the purpose
 			x[i] = position[i];
 			vx[i] = vx_val;
 		}
+#ifdef HAVE_PYTHON
+		// plotting-only setup remains here
 #endif
 	}
@@
 	// plot the temperature and vx profile along the x axis at the center
 	if (amrex::ParallelDescriptor::IOProcessor()) {
-#ifdef HAVE_PYTHON
 		double v_value_norm = 0.0;
 		double v_err_norm = 0.0;
 		double T_value_norm = 0.0;
@@
 			if (!(v_rel_err_norm < v_rel_err_tol) || !(T_rel_err_norm < T_rel_err_tol)) {
 				status = 1;
 			}
 
+#ifdef HAVE_PYTHON
 			matplotlibcpp::clf();
 			matplotlibcpp::plot(x, T, {{"label", "base"}, {"color", "C0"}});
@@
 			matplotlibcpp::title(std::format("time t = {:.4g}", sim2.tNew_[0]));
 			matplotlibcpp::save(std::format("sn_velocity_profile_n0_{:.1g}.pdf", n_amb, boost_vel_x));
 #endif
 	}
```
