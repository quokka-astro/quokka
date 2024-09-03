#ifndef RAD_TRANSPORT_HPP_
#define RAD_TRANSPORT_HPP_

#include "radiation/radiation_base.hpp"

template <typename problem_t>
void RadSystem<problem_t>::ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange)
{
	// keep radiation energy density as-is
	// convert (Fx,Fy,Fz) into reduced flux components (fx,fy,fx):
	//   F_x -> F_x / (c*E_r)

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// add reduced fluxes for each radiation group
		for (int g = 0; g < nGroups_; ++g) {
			const auto E_r = cons(i, j, k, radEnergy_index + numRadVars_ * g);
			const auto Fx = cons(i, j, k, x1RadFlux_index + numRadVars_ * g);
			const auto Fy = cons(i, j, k, x2RadFlux_index + numRadVars_ * g);
			const auto Fz = cons(i, j, k, x3RadFlux_index + numRadVars_ * g);

			// check admissibility of states
			AMREX_ASSERT(E_r > 0.0); // NOLINT

			primVar(i, j, k, primRadEnergy_index + numRadVars_ * g) = E_r;
			primVar(i, j, k, x1ReducedFlux_index + numRadVars_ * g) = Fx / (c_light_ * E_r);
			primVar(i, j, k, x2ReducedFlux_index + numRadVars_ * g) = Fy / (c_light_ * E_r);
			primVar(i, j, k, x3ReducedFlux_index + numRadVars_ * g) = Fz / (c_light_ * E_r);
		}
	});
}

template <typename problem_t>
void RadSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const & /*cons*/, array_t &maxSignal, amrex::Box const &indexRange)
{
	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double signal_max = c_hat_;
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool
{
	// check if the state variable 'cons' is a valid state
	bool isValid = true;
	for (int g = 0; g < nGroups_; ++g) {
		const auto E_r = cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fx = cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fy = cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fz = cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_];

		const auto Fnorm = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
		const auto f = Fnorm / (c_light_ * E_r);

		bool isNonNegative = (E_r > 0.);
		bool isFluxCausal = (f <= 1.);
		isValid = (isValid && isNonNegative && isFluxCausal);
	}
	return isValid;
}

template <typename problem_t> AMREX_GPU_DEVICE void RadSystem<problem_t>::amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons)
{
	// amend the state variable 'cons' to be a valid state
	for (int g = 0; g < nGroups_; ++g) {
		auto E_r = cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_];
		if (E_r < Erad_floor_) {
			E_r = Erad_floor_;
			cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_] = Erad_floor_;
		}
		const auto Fx = cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fy = cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fz = cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		if (Fx * Fx + Fy * Fy + Fz * Fz > c_light_ * c_light_ * E_r * E_r) {
			const auto Fnorm = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
			cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fx / Fnorm * c_light_ * E_r;
			cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fy / Fnorm * c_light_ * E_r;
			cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fz / Fnorm * c_light_ * E_r;
		}
	}
}

template <typename problem_t>
void RadSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				       amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	const auto dx = dx_in[0];
	const auto x1Flux = fluxArray[0];
	// const auto x1FluxDiffusive = fluxDiffusiveArray[0];
#if (AMREX_SPACEDIM >= 2)
	const auto dy = dx_in[1];
	const auto x2Flux = fluxArray[1];
	// const auto x2FluxDiffusive = fluxDiffusiveArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	const auto dz = dx_in[2];
	const auto x3Flux = fluxArray[2];
	// const auto x3FluxDiffusive = fluxDiffusiveArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		std::array<amrex::Real, nvarHyperbolic_> cons{};

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			cons[n] = consVarOld(i, j, k, nstartHyperbolic_ + n) + (AMREX_D_TERM((dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
											     +(dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
											     +(dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))));
		}

		if (!isStateValid(cons)) {
			amendRadState(cons);
		}
		AMREX_ASSERT(isStateValid(cons));

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			consVarNew(i, j, k, nstartHyperbolic_ + n) = cons[n];
		}
	});
}

template <typename problem_t>
void RadSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArrayOld*/,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
					amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	const auto dx = dx_in[0];
	const auto x1FluxOld = fluxArrayOld[0];
	const auto x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
	const auto dy = dx_in[1];
	const auto x2FluxOld = fluxArrayOld[1];
	const auto x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	const auto dz = dx_in[2];
	const auto x3FluxOld = fluxArrayOld[2];
	const auto x3Flux = fluxArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		std::array<amrex::Real, nvarHyperbolic_> cons_new{};

		// y^n+1 = (1 - a32) y^n + a32 y^(2) + dt * (0.5 - a32) * s(y^n) + dt * 0.5 * s(y^(2)) + dt * (1 - a32) * f(y^n+1)          // the last term is
		// implicit and not used here
		for (int n = 0; n < nvarHyperbolic_; ++n) {
			const double U_0 = U0(i, j, k, nstartHyperbolic_ + n);
			const double U_1 = U1(i, j, k, nstartHyperbolic_ + n);
			const double FxU_0 = (dt / dx) * (x1FluxOld(i, j, k, n) - x1FluxOld(i + 1, j, k, n));
			const double FxU_1 = (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n));
#if (AMREX_SPACEDIM >= 2)
			const double FyU_0 = (dt / dy) * (x2FluxOld(i, j, k, n) - x2FluxOld(i, j + 1, k, n));
			const double FyU_1 = (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n));
#endif
#if (AMREX_SPACEDIM == 3)
			const double FzU_0 = (dt / dz) * (x3FluxOld(i, j, k, n) - x3FluxOld(i, j, k + 1, n));
			const double FzU_1 = (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n));
#endif
			// save results in cons_new
			cons_new[n] = (1.0 - IMEX_a32) * U_0 + IMEX_a32 * U_1 + ((0.5 - IMEX_a32) * (AMREX_D_TERM(FxU_0, +FyU_0, +FzU_0))) +
				      (0.5 * (AMREX_D_TERM(FxU_1, +FyU_1, +FzU_1)));
		}

		if (!isStateValid(cons_new)) {
			amendRadState(cons_new);
		}
		AMREX_ASSERT(isStateValid(cons_new));

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			U_new(i, j, k, nstartHyperbolic_ + n) = cons_new[n];
		}
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto
RadSystem<problem_t>::ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i,
					      int j, int k, const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries) -> quokka::valarray<double, nGroups_>
{
	// compute interface-averaged cell optical depth

	// [By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.]

	// piecewise-constant reconstruction
	const double rho_L = consVar(i - 1, j, k, gasDensity_index);
	const double rho_R = consVar(i, j, k, gasDensity_index);

	const double x1GasMom_L = consVar(i - 1, j, k, x1GasMomentum_index);
	const double x1GasMom_R = consVar(i, j, k, x1GasMomentum_index);

	const double x2GasMom_L = consVar(i - 1, j, k, x2GasMomentum_index);
	const double x2GasMom_R = consVar(i, j, k, x2GasMomentum_index);

	const double x3GasMom_L = consVar(i - 1, j, k, x3GasMomentum_index);
	const double x3GasMom_R = consVar(i, j, k, x3GasMomentum_index);

	const double Egas_L = consVar(i - 1, j, k, gasEnergy_index);
	const double Egas_R = consVar(i, j, k, gasEnergy_index);

	auto massScalars_L = RadSystem<problem_t>::ComputeMassScalars(consVar, i - 1, j, k);
	auto massScalars_R = RadSystem<problem_t>::ComputeMassScalars(consVar, i, j, k);

	double Eint_L = NAN;
	double Eint_R = NAN;
	double Tgas_L = NAN;
	double Tgas_R = NAN;

	if constexpr (gamma_ != 1.0) {
		Eint_L = RadSystem<problem_t>::ComputeEintFromEgas(rho_L, x1GasMom_L, x2GasMom_L, x3GasMom_L, Egas_L);
		Eint_R = RadSystem<problem_t>::ComputeEintFromEgas(rho_R, x1GasMom_R, x2GasMom_R, x3GasMom_R, Egas_R);
		Tgas_L = quokka::EOS<problem_t>::ComputeTgasFromEint(rho_L, Eint_L, massScalars_L);
		Tgas_R = quokka::EOS<problem_t>::ComputeTgasFromEint(rho_R, Eint_R, massScalars_R);
	}

	double dl = NAN;
	if constexpr (DIR == FluxDir::X1) {
		dl = dx[0];
	} else if constexpr (DIR == FluxDir::X2) {
		dl = dx[1];
	} else if constexpr (DIR == FluxDir::X3) {
		dl = dx[2];
	}

	quokka::valarray<double, nGroups_> optical_depths{};
	if constexpr (nGroups_ == 1) {
		const double tau_L = dl * rho_L * RadSystem<problem_t>::ComputeFluxMeanOpacity(rho_L, Tgas_L);
		const double tau_R = dl * rho_R * RadSystem<problem_t>::ComputeFluxMeanOpacity(rho_R, Tgas_R);
		optical_depths[0] = (tau_L * tau_R * 2.) / (tau_L + tau_R); // harmonic mean. Alternative: 0.5*(tau_L + tau_R)
	} else {
		const auto opacity_L = DefineOpacityExponentsAndLowerValues(group_boundaries, rho_L, Tgas_L);
		const auto opacity_R = DefineOpacityExponentsAndLowerValues(group_boundaries, rho_R, Tgas_R);
		const auto tau_L = dl * rho_L * ComputeBinCenterOpacity(group_boundaries, opacity_L);
		const auto tau_R = dl * rho_R * ComputeBinCenterOpacity(group_boundaries, opacity_R);
		optical_depths = (tau_L * tau_R * 2.) / (tau_L + tau_R); // harmonic mean. Alternative: 0.5*(tau_L + tau_R)
	}

	return optical_depths;
}

template <typename problem_t>
template <FluxDir DIR>
void RadSystem<problem_t>::ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
					 amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool const use_wavespeed_correction)
{
	quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in);
	quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in);
	quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in);
	quokka::Array4View<amrex::Real, DIR> x1FluxDiffusive(x1FluxDiffusive_in);
	quokka::Array4View<const amrex::Real, DIR> consVar(consVar_in);

	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// interface-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in) {
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		amrex::GpuArray<double, nGroups_ + 1> radBoundaries_g_copy{};
		for (int g = 0; g < nGroups_ + 1; ++g) {
			radBoundaries_g_copy[g] = radBoundaries_g[g];
		}

		// HLL solver following Toro (1998) and Balsara (2017).
		// Radiation eigenvalues from Skinner & Ostriker (2013).

		// calculate cell optical depth for each photon group
		// Similar to the asymptotic-preserving flux correction in Skinner et al. (2019). Use optionally apply it here to reduce odd-even instability.
		quokka::valarray<double, nGroups_> tau_cell{};
		if (use_wavespeed_correction) {
			tau_cell = ComputeCellOpticalDepth<DIR>(consVar, dx, i, j, k, radBoundaries_g_copy);
		}

		// gather left- and right- state variables
		for (int g = 0; g < nGroups_; ++g) {
			double erad_L = x1LeftState(i, j, k, primRadEnergy_index + numRadVars_ * g);
			double erad_R = x1RightState(i, j, k, primRadEnergy_index + numRadVars_ * g);

			double fx_L = x1LeftState(i, j, k, x1ReducedFlux_index + numRadVars_ * g);
			double fx_R = x1RightState(i, j, k, x1ReducedFlux_index + numRadVars_ * g);

			double fy_L = x1LeftState(i, j, k, x2ReducedFlux_index + numRadVars_ * g);
			double fy_R = x1RightState(i, j, k, x2ReducedFlux_index + numRadVars_ * g);

			double fz_L = x1LeftState(i, j, k, x3ReducedFlux_index + numRadVars_ * g);
			double fz_R = x1RightState(i, j, k, x3ReducedFlux_index + numRadVars_ * g);

			// compute scalar reduced flux f
			double f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
			double f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);

			// Compute "un-reduced" Fx, Fy, Fz
			double Fx_L = fx_L * (c_light_ * erad_L);
			double Fx_R = fx_R * (c_light_ * erad_R);

			double Fy_L = fy_L * (c_light_ * erad_L);
			double Fy_R = fy_R * (c_light_ * erad_R);

			double Fz_L = fz_L * (c_light_ * erad_L);
			double Fz_R = fz_R * (c_light_ * erad_R);

			// check that states are physically admissible; if not, use first-order
			// reconstruction
			if ((erad_L <= 0.) || (erad_R <= 0.) || (f_L >= 1.) || (f_R >= 1.)) {
				erad_L = consVar(i - 1, j, k, radEnergy_index + numRadVars_ * g);
				erad_R = consVar(i, j, k, radEnergy_index + numRadVars_ * g);

				Fx_L = consVar(i - 1, j, k, x1RadFlux_index + numRadVars_ * g);
				Fx_R = consVar(i, j, k, x1RadFlux_index + numRadVars_ * g);

				Fy_L = consVar(i - 1, j, k, x2RadFlux_index + numRadVars_ * g);
				Fy_R = consVar(i, j, k, x2RadFlux_index + numRadVars_ * g);

				Fz_L = consVar(i - 1, j, k, x3RadFlux_index + numRadVars_ * g);
				Fz_R = consVar(i, j, k, x3RadFlux_index + numRadVars_ * g);

				// compute primitive variables
				fx_L = Fx_L / (c_light_ * erad_L);
				fx_R = Fx_R / (c_light_ * erad_R);

				fy_L = Fy_L / (c_light_ * erad_L);
				fy_R = Fy_R / (c_light_ * erad_R);

				fz_L = Fz_L / (c_light_ * erad_L);
				fz_R = Fz_R / (c_light_ * erad_R);

				f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
				f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);
			}

			// ComputeRadPressure returns F_L_and_S_L or F_R_and_S_R
			auto [F_L, S_L] = ComputeRadPressure<DIR>(erad_L, Fx_L, Fy_L, Fz_L, fx_L, fy_L, fz_L);
			S_L *= -1.; // speed sign is -1
			auto [F_R, S_R] = ComputeRadPressure<DIR>(erad_R, Fx_R, Fy_R, Fz_R, fx_R, fy_R, fz_R);

			// correct for reduced speed of light
			F_L[0] *= c_hat_ / c_light_;
			F_R[0] *= c_hat_ / c_light_;
			for (int n = 1; n < numRadVars_; ++n) {
				F_L[n] *= c_hat_ * c_light_;
				F_R[n] *= c_hat_ * c_light_;
			}
			S_L *= c_hat_;
			S_R *= c_hat_;

			const quokka::valarray<double, numRadVars_> U_L = {erad_L, Fx_L, Fy_L, Fz_L};
			const quokka::valarray<double, numRadVars_> U_R = {erad_R, Fx_R, Fy_R, Fz_R};

			// Adjusting wavespeeds is no longer necessary with the IMEX PD-ARS scheme.
			// Read more in https://github.com/quokka-astro/quokka/pull/582
			// However, we let the user optionally apply it to reduce odd-even instability.
			quokka::valarray<double, numRadVars_> epsilon = {1.0, 1.0, 1.0, 1.0};
			if (use_wavespeed_correction) {
				// no correction for odd zones
				if ((i + j + k) % 2 == 0) {
					const double S_corr = std::min(1.0, 1.0 / tau_cell[g]); // Skinner et al.
					epsilon = {S_corr, 1.0, 1.0, 1.0};			// Skinner et al. (2019)
				}
			}

			AMREX_ASSERT(std::abs(S_L) <= c_hat_); // NOLINT
			AMREX_ASSERT(std::abs(S_R) <= c_hat_); // NOLINT

			// in the frozen Eddington tensor approximation, we are always
			// in the star region, so F = F_star
			const quokka::valarray<double, numRadVars_> F =
			    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R + epsilon * (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

			// check states are valid
			AMREX_ASSERT(!std::isnan(F[0])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[1])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[2])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[3])); // NOLINT

			x1Flux(i, j, k, radEnergy_index + numRadVars_ * g - nstartHyperbolic_) = F[0];
			x1Flux(i, j, k, x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[1];
			x1Flux(i, j, k, x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[2];
			x1Flux(i, j, k, x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[3];

			const quokka::valarray<double, numRadVars_> diffusiveF =
			    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R + (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

			x1FluxDiffusive(i, j, k, radEnergy_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[0];
			x1FluxDiffusive(i, j, k, x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[1];
			x1FluxDiffusive(i, j, k, x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[2];
			x1FluxDiffusive(i, j, k, x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[3];
		} // end loop over radiation groups
	});
}

#endif // RAD_TRANSPORT_HPP_