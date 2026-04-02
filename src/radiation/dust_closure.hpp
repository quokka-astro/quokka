// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef DUST_CLOSURE_HPP_
#define DUST_CLOSURE_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

/// Select dust model based on coupling strength.
/// Returns gas_only when enable_dust_gas_thermal_coupling_model_ is false (constexpr path).
/// Otherwise returns coupled or decoupled based on threshold.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SelectDustModel(double T_gas0, double T_d0, double Egas0, double coeff_n) -> DustModel
{
	if constexpr (!enable_dust_gas_thermal_coupling_model_) {
		return DustModel::gas_only;
	} else {
		const double cscale = c_light_ / c_hat_;
		const double max_Gamma_gd = coeff_n * std::max(std::sqrt(T_gas0) * T_gas0, std::sqrt(T_d0) * T_d0);
		if (cscale * max_Gamma_gd < ISM_Traits<problem_t>::gas_dust_coupling_threshold * Egas0) {
			return DustModel::decoupled;
		}
		return DustModel::coupled;
	}
}

/// Compute dust temperature from the current Newton iterate.
/// Only called for dust path (coupled or decoupled). Gas-only sets Td = T_gas in caller.
///
/// - coupled: Td = T_gas - sum(R_all) / (Nd * sqrt(T_gas))
///   where R_all sums over ALL groups (thermal + chemical), because chemical-band
///   photons absorbed by dust heat the dust.
/// - decoupled: at n==0, use T_d0; thereafter, Td is updated by the Newton step
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeDustTemperatureFromIterate(DustModel model, double T_gas,
									      quokka::valarray<double, nGroups_> const &Rvec_all, double coeff_n,
									      double T_d0, int newton_iter, double tempFloor) -> double
{
	// When enable_dust_gas_thermal_coupling_model_ is true, model is always
	// coupled or decoupled (never gas_only), so no need to check for gas_only here.
	double T_d = NAN;

	if (model == DustModel::coupled) {
		if (newton_iter == 0) {
			T_d = T_d0;
		} else {
			// sum over ALL groups (thermal + chemical)
			T_d = T_gas - sum(Rvec_all) / (coeff_n * std::sqrt(T_gas));
		}
	} else { // decoupled
		if (newton_iter == 0) {
			T_d = T_d0;
		}
		// For decoupled model at newton_iter > 0, T_d is updated by the Newton
		// step (delta_x is applied to T_d directly). This is handled in the caller.
	}

	// Enforce dust temperature floor
	if (T_d < tempFloor) {
		T_d = tempFloor;
	}

	return T_d;
}

#endif // DUST_CLOSURE_HPP_
