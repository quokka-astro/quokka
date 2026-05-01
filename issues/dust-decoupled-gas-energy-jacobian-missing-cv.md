# Decoupled dust gas-energy Newton solve uses the wrong cooling Jacobian

Severity: High

## Bug

`SolveGasDustRadiationEnergyExchange` and `SolveGasDustRadiationEnergyExchangeWithPE` use `BackwardEulerOneVariable` to solve for `Egas_guess` in the decoupled dust model. Their residual functions are written in terms of gas internal energy, but the Jacobian lambdas add `sum(DefineNetCoolingRateTempDerivative(...))` directly:

```cpp
return 1.0 + sum(d_cooling_d_Tgas_);
```

`DefineNetCoolingRateTempDerivative` is a derivative with respect to gas temperature. Since the Newton variable is gas internal energy, this term must be multiplied by `dT/dE = 1 / c_v`. The coupled Jacobian in the same file already applies that conversion via `sum(cooling_derivative) / c_v`.

Without the conversion, the Newton step has units and magnitude wrong. In cooling-dominated decoupled dust cells this can converge slowly, converge to an inaccurate update under the relative-change stop, or fail and return `-1.0`, feeding a negative internal energy back to the radiation source update path.

## Patch

```diff
diff --git a/src/radiation/radiation_dust_system.hpp b/src/radiation/radiation_dust_system.hpp
--- a/src/radiation/radiation_dust_system.hpp
+++ b/src/radiation/radiation_dust_system.hpp
@@
 		auto jac = [=](double Egas_) -> double {
 			const double T_gas_ = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
 			const auto d_cooling_d_Tgas_ = DefineNetCoolingRateTempDerivative(T_gas_, H_num_den) * dt;
-			return 1.0 + sum(d_cooling_d_Tgas_);
+			const double c_v_ = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas_, massScalars);
+			return 1.0 + sum(d_cooling_d_Tgas_) / c_v_;
 		};
@@
 		auto jac = [=](double Egas_) -> double {
 			const double T_gas_ = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
 			const auto d_cooling_d_Tgas_ = DefineNetCoolingRateTempDerivative(T_gas_, H_num_den) * dt;
-			return 1.0 + sum(d_cooling_d_Tgas_);
+			const double c_v_ = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas_, massScalars);
+			return 1.0 + sum(d_cooling_d_Tgas_) / c_v_;
 		};
```
