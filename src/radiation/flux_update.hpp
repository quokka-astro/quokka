// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef FLUX_UPDATE_HPP_ // NOLINT
#define FLUX_UPDATE_HPP_

/// \file flux_update.hpp
/// \brief Radiation flux relaxation, gas momentum update, and work term computation.
///
/// Contains:
///   - UpdateFluxAndMomentum: updates radiation flux and gas momentum for all groups
///   - ComputeWorkTerm: computes the work term from updated flux and momentum
///   - WorkConverged: checks convergence of the outer work-lag iteration

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

/// Update radiation flux and gas momentum for all groups.
/// For beta_order_ == 0: simple exponential damping.
/// For beta_order_ == 1: includes Planck and pressure terms, plus work term handling.
/// Also updates energy.Egas and energy.work when include_work_term_in_source is true.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::UpdateFluxAndMomentum(int const i, int const j, int const k, arrayconst_t const &consPrev,
								  NewtonIterationResult<problem_t> &energy, double const dt,
								  double const gas_update_factor, double const Ekin0) -> FluxUpdateResult<problem_t>
{
	amrex::GpuArray<amrex::Real, 3> Frad_t0{};
	amrex::GpuArray<amrex::Real, 3> dMomentum{0., 0., 0.};
	amrex::GpuArray<amrex::GpuArray<amrex::Real, nGroups_>, 3> Frad_t1{};

	// make a copy of radBoundaries_
	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	double const rho = consPrev(i, j, k, gasDensity_index);
	const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
	const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
	const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
	const std::array<double, 3> gasMtm0 = {x1GasMom0, x2GasMom0, x3GasMom0};

	auto const fourPiBoverC = ComputeThermalRadiationMultiGroup(energy.T_d, radBoundaries_g);
	auto const kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g, rho, energy.T_d);

	const double chat = c_hat_;

	for (int g = 0; g < nGroups_; ++g) {
		Frad_t0[0] = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
		Frad_t0[1] = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
		Frad_t0[2] = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);

		if constexpr ((gamma_ == 1.0) || (beta_order_ == 0)) {
			for (int n = 0; n < 3; ++n) {
				Frad_t1[n][g] = Frad_t0[n] / (1.0 + rho * energy.opacity_terms.kappaF[g] * chat * dt);
				// Compute conservative gas momentum update
				dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c_light_ * chat);
			}
		} else {
			const auto erad = energy.EradVec[g];
			std::array<double, 3> v_terms{};

			auto fx = Frad_t0[0] / (c_light_ * erad);
			auto fy = Frad_t0[1] / (c_light_ * erad);
			auto fz = Frad_t0[2] / (c_light_ * erad);
			double F_coeff = chat * rho * energy.opacity_terms.kappaF[g] * dt;
			auto Tedd = ComputeEddingtonTensor(fx, fy, fz);

			for (int n = 0; n < 3; ++n) {
				// compute thermal radiation term
				double Planck_term = NAN;

				if constexpr (include_delta_B) {
					Planck_term =
					    energy.opacity_terms.kappaP[g] * fourPiBoverC[g] - 1.0 / 3.0 * energy.opacity_terms.delta_nu_kappa_B_at_edge[g];
				} else {
					Planck_term = energy.opacity_terms.kappaP[g] * fourPiBoverC[g];
				}

				Planck_term *= chat * dt * gasMtm0[n];

				// compute radiation pressure
				double pressure_term = 0.0;
				for (int z = 0; z < 3; ++z) {
					pressure_term += gasMtm0[z] * Tedd[n][z] * erad;
				}
				// Simplification: assuming Eddington tensors are the same for all groups, we have kappaP = kappaE
				if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
					pressure_term *= chat * dt * energy.opacity_terms.kappaE[g];
				} else {
					pressure_term *= chat * dt * (1.0 + kappa_expo_and_lower_value[0][g]) * energy.opacity_terms.kappaE[g];
				}

				v_terms[n] = Planck_term + pressure_term;
			}

			for (int n = 0; n < 3; ++n) {
				// Compute flux update
				Frad_t1[n][g] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

				// Compute conservative gas momentum update
				dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c_light_ * chat);
			}
		}
	}

	amrex::Real x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0];
	amrex::Real x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1];
	amrex::Real x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2];

	FluxUpdateResult<problem_t> updated_flux;

	for (int g = 0; g < nGroups_; ++g) {
		updated_flux.Erad[g] = energy.EradVec[g];
	}

	// 3. Deal with the work term.
	if constexpr ((gamma_ != 1.0) && (beta_order_ == 1)) {
		// compute difference in gas kinetic energy before and after momentum update
		amrex::Real const Egastot1 = ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, energy.Egas);
		amrex::Real const Ekin1 = Egastot1 - energy.Egas;
		amrex::Real const dEkin_work = Ekin1 - Ekin0;

		if constexpr (include_work_term_in_source) {
			// New scheme: the work term is included in the source terms. The work done by radiation went to internal energy, but it
			// should go to the kinetic energy. Remove the work term from internal energy.
			energy.Egas -= dEkin_work;
			// The work term is included in the source term, but it is lagged. We update the work term here.
			for (int g = 0; g < nGroups_; ++g) {
				// compute new work term from the updated radiation flux and velocity
				// work = v * F * chi
				if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
					energy.work[g] = (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]) *
							 energy.opacity_terms.kappaF[g] * chat / (c_light_ * c_light_) * dt;
				} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum ||
						     opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
					energy.work[g] = (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]) *
							 (1.0 + kappa_expo_and_lower_value[0][g]) * energy.opacity_terms.kappaF[g] * chat /
							 (c_light_ * c_light_) * dt;
				}
			}
		} else {
			// Old scheme: the source term does not include the work term, so we add the work term to the Erad.

			// compute loss of radiation energy to gas kinetic energy
			auto dErad_work = -(c_hat_ / c_light_) * dEkin_work;

			// apportion dErad_work according to kappaF_i * (v * F_i)
			quokka::valarray<double, nGroups_> energyLossFractions{};
			if constexpr (nGroups_ == 1) {
				energyLossFractions[0] = 1.0;
			} else {
				// compute energyLossFractions
				for (int g = 0; g < nGroups_; ++g) {
					energyLossFractions[g] = energy.opacity_terms.kappaF[g] *
								 (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]);
				}
				auto energyLossFractionsTot = sum(energyLossFractions);
				if (energyLossFractionsTot != 0.0) {
					energyLossFractions /= energyLossFractionsTot;
				} else {
					energyLossFractions.fillin(0.0);
				}
			}
			for (int g = 0; g < nGroups_; ++g) {
				auto radEnergyNew = energy.EradVec[g] + dErad_work * energyLossFractions[g];
				// AMREX_ASSERT(radEnergyNew > 0.0);
				if (radEnergyNew < Erad_floor_) {
					// return energy to Egas_guess
					energy.Egas -= (Erad_floor_ - radEnergyNew) * (c_light_ / c_hat_);
					radEnergyNew = Erad_floor_;
				}
				updated_flux.Erad[g] = radEnergyNew;
			}
		}
	}

	x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0] * gas_update_factor;
	x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1] * gas_update_factor;
	x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2] * gas_update_factor;
	updated_flux.gasMomentum = {x1GasMom1, x2GasMom1, x3GasMom1};
	updated_flux.Frad = Frad_t1;

	return updated_flux;
}

/// Compute the work term from updated flux and momentum.
/// work[g] = (v . F[g]) * kappaF[g] * chat / c^2 * dt
/// This is extracted from UpdateFluxAndMomentum for use in isolation when only the work term is needed.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeWorkTerm(amrex::GpuArray<amrex::Real, 3> const &gasMomentum,
							    amrex::GpuArray<amrex::GpuArray<amrex::Real, nGroups_>, 3> const &Frad,
							    OpacityTerms<problem_t> const &opacity_terms, double const dt)
    -> quokka::valarray<double, nGroups_>
{
	// make a copy of radBoundaries_
	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	const double chat = c_hat_;
	quokka::valarray<double, nGroups_> work{};

	// Only meaningful for beta_order_ == 1 with work term in source
	if constexpr ((gamma_ != 1.0) && (beta_order_ == 1) && include_work_term_in_source) {
		for (int g = 0; g < nGroups_; ++g) {
			const double vdotF = gasMomentum[0] * Frad[0][g] + gasMomentum[1] * Frad[1][g] + gasMomentum[2] * Frad[2][g];
			if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
				work[g] = vdotF * opacity_terms.kappaF[g] * chat / (c_light_ * c_light_) * dt;
			} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum ||
					     opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
				// NOTE: kappa_expo_and_lower_value is not available here. This standalone function
				// is a simplified version for piecewise_constant_opacity. For PPL models, use
				// UpdateFluxAndMomentum which computes the full work term internally.
				work[g] = vdotF * opacity_terms.kappaF[g] * chat / (c_light_ * c_light_) * dt;
			}
		}
	}

	return work;
}

/// Check convergence of the work term in the outer work-lag iteration.
/// Returns true if the work term has converged or if work terms are not used.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::WorkConverged(quokka::valarray<double, nGroups_> const &work,
							  quokka::valarray<double, nGroups_> const &work_prev, double const rho,
							  amrex::GpuArray<double, 3> const &gasMomentum, double const Egas_guess) -> bool
{
	if constexpr ((beta_order_ == 0) || (gamma_ == 1.0) || (!include_work_term_in_source)) {
		return true;
	} else {
		auto const Egastot1 = ComputeEgasFromEint(rho, gasMomentum[0], gasMomentum[1], gasMomentum[2], Egas_guess);
		const double rel_lag_tol = 1.0e-8;
		const double lag_tol = 1.0e-13;
		double ref_work = rel_lag_tol * sum(abs(work));
		ref_work = std::max(ref_work, lag_tol * Egastot1 / (c_light_ / c_hat_));
		if (sum(abs(work - work_prev)) > ref_work) {
			return false;
		}
		return true;
	}
}

#endif // FLUX_UPDATE_HPP_
