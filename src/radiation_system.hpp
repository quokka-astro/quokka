#ifndef RADIATION_SYSTEM_HPP_ // NOLINT
#define RADIATION_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file radiation_system.hpp
/// \brief Defines a class for solving the (1d) radiation moment equations.
///

// c++ headers
#include <cmath>

// library headers

// internal headers
#include "AMReX_Array.H"
#include "AMReX_REAL.H"
#include "ArrayView.hpp"
#include "hyperbolic_system.hpp"
#include "simulation.hpp"
#include "valarray.hpp"

// physical constants in CGS units
static constexpr double c_light_cgs_ = 2.99792458e10;		// cgs
static constexpr double radiation_constant_cgs_ = 7.5646e-15;	// cgs
static constexpr double hydrogen_mass_cgs_ = 1.6726231e-24;	// cgs
static constexpr double boltzmann_constant_cgs_ = 1.380658e-16; // cgs

// this struct is specialized by the user application code
//
template <typename problem_t> struct RadSystem_Traits {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = c_light_cgs_;
	static constexpr double radiation_constant = radiation_constant_cgs_;
	static constexpr double mean_molecular_mass = hydrogen_mass_cgs_;
	static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
	static constexpr double gamma = 5. / 3.;

	static constexpr double Erad_floor = 0.;

	static constexpr bool do_marshak_left_boundary = false;
	static constexpr double T_marshak_left = 0.;
};

/// Class for the radiation moment equations
///
template <typename problem_t> class RadSystem : public HyperbolicSystem<problem_t>
{
      public:
	enum consVarIndex {
		radEnergy_index = 0,
		x1RadFlux_index = 1,
		x2RadFlux_index = 2,
		x3RadFlux_index = 3,
		gasEnergy_index = 4,
		gasDensity_index = 5,
		x1GasMomentum_index = 6,
		x2GasMomentum_index = 7,
		x3GasMomentum_index = 8,
		consVarEnumEnd = 9
	};

	enum primVarIndex {
		primRadEnergy_index = 0,
		x1ReducedFlux_index = 1,
		x2ReducedFlux_index = 2,
		x3ReducedFlux_index = 3,
		primVarEnumEnd = 4
	};

	// C++ standard does not allow constexpr to be uninitialized, even in a templated class!
	static constexpr double c_light_ = RadSystem_Traits<problem_t>::c_light;
	static constexpr double c_hat_ = RadSystem_Traits<problem_t>::c_hat;
	static constexpr double radiation_constant_ =
	    RadSystem_Traits<problem_t>::radiation_constant;
	static constexpr double mean_molecular_mass_ =
	    RadSystem_Traits<problem_t>::mean_molecular_mass;
	static constexpr double boltzmann_constant_ =
	    RadSystem_Traits<problem_t>::boltzmann_constant;
	static constexpr double gamma_ = RadSystem_Traits<problem_t>::gamma;

	static constexpr double Erad_floor_ = RadSystem_Traits<problem_t>::Erad_floor;

	static constexpr bool do_marshak_left_boundary_ =
	    RadSystem_Traits<problem_t>::do_marshak_left_boundary;
	static constexpr double T_marshak_left_ = RadSystem_Traits<problem_t>::T_marshak_left;

	// static functions

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons,
					  array_t &maxSignal, amrex::Box const &indexRange);
	static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons,
					 array_t &primVar, amrex::Box const &indexRange);

	template <FluxDir DIR>
	static void ComputeFluxes(array_t &x1Flux,
				  amrex::Array4<const amrex::Real> const &x1LeftState,
				  amrex::Array4<const amrex::Real> const &x1RightState,
				  amrex::Box const &indexRange, arrayconst_t &consVar,
				  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);

	// static void ComputeFirstOrderFluxes(amrex::Array4<const amrex::Real> const &consVar,
	//					   array_t &x1FluxDiffusive,
	//					   amrex::Box const &indexRange);

	static void SetRadEnergySource(array_t &radEnergy, amrex::Box const &indexRange,
				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
				       amrex::Real time);

	static void AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource,
				   arrayconst_t &advectionFluxes, amrex::Box const &indexRange,
				   amrex::Real dt);
	static void ComputeSourceTermsExplicit(arrayconst_t &consPrev,
					       arrayconst_t &radEnergySource, array_t &src,
					       amrex::Box const &indexRange, amrex::Real dt);

	static auto ComputeEddingtonFactor(double f) -> double;
	static auto ComputeOpacity(double rho, double Tgas) -> double;
	static auto ComputeOpacityTempDerivative(double rho, double Tgas) -> double;
	static auto ComputeTgasFromEgas(double rho, double Egas) -> double;
	static auto ComputeEgasFromTgas(double rho, double Tgas) -> double;
	static auto ComputeEgasTempDerivative(double rho, double Tgas) -> double;
	static auto ComputeEintFromEgas(double density, double X1GasMom, double X2GasMom,
					double X3GasMom, double Etot) -> double;
	static auto ComputeEgasFromEint(double density, double X1GasMom, double X2GasMom,
					double X3GasMom, double Eint) -> double;

	static auto ComputeCellOpticalDepth(arrayconst_t &consVar,
					    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i,
					    int j, int k) -> double;

	// requires GPU reductions
	// auto CheckStatesValid(array_t &cons, std::pair<int, int> range) -> bool;
};

template <typename problem_t>
void RadSystem<problem_t>::SetRadEnergySource(array_t &radEnergySource,
					      amrex::Box const &indexRange,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					      amrex::Real time)
{
	// do nothing -- user implemented
}

template <typename problem_t>
void RadSystem<problem_t>::ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons,
						array_t &primVar, amrex::Box const &indexRange)
{
	// keep radiation energy density as-is
	// convert (Fx,Fy,Fz) into reduced flux components (fx,fy,fx):
	//   F_x -> F_x / (c*E_r)

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto E_r = cons(i, j, k, radEnergy_index);
		const auto Fx = cons(i, j, k, x1RadFlux_index);
		const auto Fy = cons(i, j, k, x2RadFlux_index);
		const auto Fz = cons(i, j, k, x3RadFlux_index);
		const auto reducedFluxX1 = Fx / (c_light_ * E_r);
		const auto reducedFluxX2 = Fy / (c_light_ * E_r);
		const auto reducedFluxX3 = Fz / (c_light_ * E_r);

		// check admissibility of states
		AMREX_ASSERT(E_r > 0.0); // NOLINT

		primVar(i, j, k, primRadEnergy_index) = E_r;
		primVar(i, j, k, x1ReducedFlux_index) = reducedFluxX1;
		primVar(i, j, k, x2ReducedFlux_index) = reducedFluxX2;
		primVar(i, j, k, x3ReducedFlux_index) = reducedFluxX3;
	});
}

template <typename problem_t>
void RadSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const & /*cons*/,
						 array_t &maxSignal, amrex::Box const &indexRange)
{
	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double signal_max = c_hat_;
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeEddingtonFactor(double f_in) -> double
{
	// f is the reduced flux == |F|/cE.
	// compute Levermore (1984) closure [Eq. 25]
	// the is the M1 closure that is derived from Lorentz invariance
	const double f = clamp(f_in, 0., 1.); // restrict f to be within [0, 1]
	const double f_fac = std::sqrt(4.0 - 3.0 * (f * f));
	const double chi = (3.0 + 4.0 * (f * f)) / (5.0 + 2.0 * f_fac);

#if 0
	// compute Minerbo (1978) closure [piecewise approximation]
	// (For unknown reasons, this closure tends to work better
	// than the Levermore/Lorentz closure on the Su & Olson 1997 test.)
	const double chi = (f < 1. / 3.) ? (1. / 3.) : (0.5 - f + 1.5 * f*f);
#endif

	return chi;
}

// TODO(ben): direction dependent!!!
template <typename problem_t>
auto RadSystem<problem_t>::ComputeCellOpticalDepth(arrayconst_t &consVar,
						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						   int i, int j, int k) -> double
{
	// compute interface-averaged cell optical depth

	// [By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.]

	const double rho_L =
	    consVar(i - 1, j, k, gasDensity_index); // piecewise-constant reconstruction
	const double rho_R = consVar(i, j, k, gasDensity_index);

	const double x1GasMom_L = consVar(i - 1, j, k, x1GasMomentum_index);
	const double x1GasMom_R = consVar(i, j, k, x1GasMomentum_index);

	const double Egas_L = consVar(i - 1, j, k, gasEnergy_index);
	const double Egas_R = consVar(i, j, k, gasEnergy_index);

	const double Eint_L = RadSystem<problem_t>::ComputeEintFromEgas(rho_L, x1GasMom_L, 0., 0.,
									Egas_L); // modify in 3d
	const double Eint_R = RadSystem<problem_t>::ComputeEintFromEgas(rho_R, x1GasMom_R, 0., 0.,
									Egas_R); // modify in 3d

	const double Tgas_L = RadSystem<problem_t>::ComputeTgasFromEgas(rho_L, Eint_L);
	const double Tgas_R = RadSystem<problem_t>::ComputeTgasFromEgas(rho_R, Eint_R);

	const double tau_L =
	    dx[0] * rho_L * RadSystem<problem_t>::ComputeOpacity(rho_L, Tgas_L); // modify in 3d
	const double tau_R =
	    dx[0] * rho_R * RadSystem<problem_t>::ComputeOpacity(rho_R, Tgas_R); // modify in 3d

	return (2.0 * tau_L * tau_R) / (tau_L + tau_R); // harmonic mean
	//	return 0.5*(tau_L + tau_R); // arithmetic mean
}

template <typename problem_t>
template <FluxDir DIR>
void RadSystem<problem_t>::ComputeFluxes(array_t &x1Flux,
					 amrex::Array4<const amrex::Real> const &x1LeftState,
					 amrex::Array4<const amrex::Real> const &x1RightState,
					 amrex::Box const &indexRange, arrayconst_t &consVar,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
	quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in);
	quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in);
	quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in);

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// interface-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in) {
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// HLL solver following Toro (1998) and Balsara (2017).
		// Radiation eigenvalues from Skinner & Ostriker (2013).

		// gather left- and right- state variables
		const double erad_L = x1LeftState(i, j, k, primRadEnergy_index);
		const double erad_R = x1RightState(i, j, k, primRadEnergy_index);

		const double fx_L = x1LeftState(i, j, k, x1ReducedFlux_index);
		const double fx_R = x1RightState(i, j, k, x1ReducedFlux_index);

		const double fy_L = x1LeftState(i, j, k, x2ReducedFlux_index);
		const double fy_R = x1LeftState(i, j, k, x2ReducedFlux_index);

		const double fz_L = x1LeftState(i, j, k, x3ReducedFlux_index);
		const double fz_R = x1LeftState(i, j, k, x3ReducedFlux_index);

		const double fvec_L[3] = {fx_L, fy_L, fz_L};
		const double fvec_R[3] = {fx_R, fy_R, fz_R};

		// compute scalar reduced flux f
		const double f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
		const double f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);

		// check that states are physically admissible
		AMREX_ASSERT(erad_L > 0.0); // NOLINT
		AMREX_ASSERT(erad_R > 0.0); // NOLINT

		// angle between interface and radiation flux \hat{n}
		// If direction is undefined, just drop direction-dependent
		// terms.
		double n_L[3];
		double n_R[3];

		for (int i = 0; i < 3; ++i) {
			n_L[i] = (f_L > 0.) ? (fvec_L[i] / f_L) : 0.;
			n_R[i] = (f_R > 0.) ? (fvec_R[i] / f_R) : 0.;
		}

		// Compute "un-reduced" Fx, Fy, Fz
		const double Fx_L = fx_L * (c_light_ * erad_L);
		const double Fx_R = fx_R * (c_light_ * erad_R);

		const double Fy_L = fy_L * (c_light_ * erad_L);
		const double Fy_R = fy_R * (c_light_ * erad_R);

		const double Fz_L = fz_L * (c_light_ * erad_L);
		const double Fz_R = fz_R * (c_light_ * erad_R);

		// compute radiation pressure tensors
		const double chi_L = RadSystem<problem_t>::ComputeEddingtonFactor(f_L);
		const double chi_R = RadSystem<problem_t>::ComputeEddingtonFactor(f_R);

		AMREX_ASSERT((chi_L >= 1. / 3.) && (chi_L <= 1.0)); // NOLINT
		AMREX_ASSERT((chi_R >= 1. / 3.) && (chi_R <= 1.0)); // NOLINT

		// diagonal term of Eddington tensor
		const double Tdiag_L = (1.0 - chi_L) / 2.0;
		const double Tdiag_R = (1.0 - chi_R) / 2.0;

		// anisotropic term of Eddington tensor (in the direction of the
		// rad. flux)
		const double Tf_L = (3.0 * chi_L - 1.0) / 2.0;
		const double Tf_R = (3.0 * chi_R - 1.0) / 2.0;

		// assemble Eddington tensor
		double T_L[3][3];
		double T_R[3][3];

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				const double delta_ij = (i == j) ? 1 : 0;
				T_L[i][j] = Tdiag_L * delta_ij + Tf_L * (n_L[i] * n_L[j]);
				T_R[i][j] = Tdiag_R * delta_ij + Tf_R * (n_R[i] * n_R[j]);
			}
		}

		// compute the elements of the total radiation pressure tensor
		double P_L[3][3];
		double P_R[3][3];

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				P_L[i][j] = T_L[i][j] * erad_L;
				P_R[i][j] = T_R[i][j] * erad_R;
			}
		}

		// asymptotic-preserving correction (new in this code) -- direction-dependent!
		const double tau_cell = ComputeCellOpticalDepth(consVar, dx, i, j, k);
		constexpr int fluxdim = 4;
		const quokka::valarray<double, fluxdim> epsilon = {
		    std::min(1.0, 1.0 / (tau_cell * tau_cell)), 1.0};

		// inspired by https://arxiv.org/pdf/2102.02212.pdf
		// ensures that signal speed -> c \sqrt{f_xx} / tau_cell in the diffusion limit
		const auto S_corr = std::min(1.0, 1.0 / tau_cell);

		// frozen Eddington tensor approximation, following Balsara
		// (1999) [JQSRT Vol. 61, No. 5, pp. 617–627, 1999], Eq. 46.
		if constexpr (DIR == FluxDir::X1) {
			Tnormal_L = T_L[0][0];
			Tnormal_R = T_R[0][0];
		} else if constexpr (DIR == FluxDir::X2) {
			Tnormal_L = T_L[1][1];
			Tnormal_R = T_R[1][1];
		} else if constexpr (DIR == FluxDir::X3) {
			Tnormal_L = T_L[2][2];
			Tnormal_R = T_R[2][2];
		}
		const double S_L = -c_hat_ * S_corr * std::sqrt(Tnormal_L);
		const double S_R = c_hat_ * S_corr * std::sqrt(Tnormal_R);

		AMREX_ASSERT(std::abs(S_L) <= c_hat_); // NOLINT
		AMREX_ASSERT(std::abs(S_R) <= c_hat_); // NOLINT

		// compute fluxes F_L, F_R
		// P_nx, P_ny, P_nz indicate components where 'n' is the direction of the face
		// normal
		// F_n is the radiation flux component in the direction of the face normal
		double Fn_L = NAN;
		double Fn_R = NAN;
		double Pnx_L = NAN;
		double Pnx_R = NAN;
		double Pny_L = NAN;
		double Pny_R = NAN;
		double Pnz_L = NAN;
		double Pnz_R = NAN;

		if constexpr (DIR == FluxDir::X1) {
			Fn_L = Fx_L;
			Fn_R = Fx_R;

			Pnx_L = P_L[0][0];
			Pny_L = P_L[0][1];
			Pnz_L = P_L[0][2];

			Pnx_R = P_R[0][0];
			Pny_R = P_R[0][1];
			Pnz_R = P_R[0][2];
		} else if constexpr (DIR == FluxDir::X2) {
			Fn_L = Fy_L;
			Fn_R = Fy_R;

			Pnx_L = P_L[1][0];
			Pny_L = P_L[1][1];
			Pnz_L = P_L[1][2];

			Pnx_R = P_R[1][0];
			Pny_R = P_R[1][1];
			Pnz_R = P_R[1][2];
		} else if constexpr (DIR == FluxDir::X3) {
			Fn_L = Fz_L;
			Fn_L = Fz_R;

			Pnx_L = P_L[2][0];
			Pny_L = P_L[2][1];
			Pnz_L = P_L[2][2];

			Pnx_R = P_R[2][0];
			Pny_R = P_R[2][1];
			Pnz_R = P_R[2][2];
		}

		const quokka::valarray<double, fluxdim> F_L = {
		    (c_hat_ / c_light_) * Fn_L, (c_hat_ * c_light_) * Pnx_L,
		    (c_hat_ * c_light_) * Pny_L, (c_hat_ * c_light_) * Pnz_L};

		const quokka::valarray<double, fluxdim> F_R = {
		    (c_hat_ / c_light_) * Fn_R, (c_hat_ * c_light_) * Pnx_R,
		    (c_hat_ * c_light_) * Pny_R, (c_hat_ * c_light_) * Pnz_R};

		const quokka::valarray<double, fluxdim> U_L = {erad_L, Fx_L, Fy_L, Fz_L};
		const quokka::valarray<double, fluxdim> U_R = {erad_R, Fx_R, Fy_R, Fz_R};

		const quokka::valarray<double, fluxdim> F_star =
		    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R +
		    epsilon * (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

		quokka::valarray<double, fluxdim> F{};

		// in the frozen Eddington tensor approximation, we are always
		// in the star region, so F = F_star
		F = F_star;

		// check states are valid
		AMREX_ASSERT(!std::isnan(F[0])); // NOLINT
		AMREX_ASSERT(!std::isnan(F[1])); // NOLINT
		AMREX_ASSERT(!std::isnan(F[2])); // NOLINT
		AMREX_ASSERT(!std::isnan(F[3])); // NOLINT

		x1Flux(i, j, k, radEnergy_index) = F[0];
		x1Flux(i, j, k, x1RadFlux_index) = F[1];
		x1Flux(i, j, k, x2RadFlux_index) = F[2];
		x1Flux(i, j, k, x3RadFlux_index) = F[3];
	});
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeOpacity(const double /*rho*/, const double /*Tgas*/) -> double
{
	return 1.0;
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeOpacityTempDerivative(const double /*rho*/, const double /*Tgas*/)
    -> double
{
	return 0.0;
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeTgasFromEgas(const double rho, const double Egas) -> double
{
	const double c_v = boltzmann_constant_ / (mean_molecular_mass_ * (gamma_ - 1.0));
	return (Egas / (rho * c_v));
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeEgasFromTgas(const double rho, const double Tgas) -> double
{
	const double c_v = boltzmann_constant_ / (mean_molecular_mass_ * (gamma_ - 1.0));
	return (rho * c_v * Tgas);
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeEgasTempDerivative(const double rho, const double /*Tgas*/)
    -> double
{
	const double c_v = boltzmann_constant_ / (mean_molecular_mass_ * (gamma_ - 1.0));
	return (rho * c_v);
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeEintFromEgas(const double density, const double X1GasMom,
					       const double X2GasMom, const double X3GasMom,
					       const double Etot) -> double
{
	const double p_sq = X1GasMom * X1GasMom + X2GasMom * X2GasMom + X3GasMom * X3GasMom;
	const double Ekin = p_sq / (2.0 * density);
	const double Eint = Etot - Ekin;
	return Eint;
}

template <typename problem_t>
auto RadSystem<problem_t>::ComputeEgasFromEint(const double density, const double X1GasMom,
					       const double X2GasMom, const double X3GasMom,
					       const double Eint) -> double
{
	const double p_sq = X1GasMom * X1GasMom + X2GasMom * X2GasMom + X3GasMom * X3GasMom;
	const double Ekin = p_sq / (2.0 * density);
	const double Etot = Eint + Ekin;
	return Etot;
}

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource,
					  arrayconst_t &advectionFluxes,
					  amrex::Box const &indexRange, amrex::Real dt)
{
	arrayconst_t &consPrev = consVar; // make read-only
	array_t &consNew = consVar;

	// Lorentz transform the radiation variables into the comoving frame
	// TransformIntoComovingFrame(fluid_velocity);

	// Add source terms

	// 1. Compute gas energy and radiation energy update following Howell &
	// Greenough [Journal of Computational Physics 184 (2003) 53–78].

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double c = c_light_;
		const double chat = c_hat_;
		const double a_rad = radiation_constant_;

		// load fluid properties
		const double rho = consPrev(i, j, k, gasDensity_index);
		const double Egastot0 = consPrev(i, j, k, gasEnergy_index);
		const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const double vx0 = x1GasMom0 / rho; // needed to update kinetic energy
		const double Egas0 =
		    ComputeEintFromEgas(rho, x1GasMom0, 0, 0, Egastot0); // modify in 3d

		// load radiation energy
		const double Erad0 = consPrev(i, j, k, radEnergy_index);

		AMREX_ASSERT(Egas0 > 0.0); // NOLINT
		AMREX_ASSERT(Erad0 > 0.0); // NOLINT

		const double Etot0 = Egas0 + (c / chat) * Erad0;

		// BEGIN NEWTON-RAPHSON LOOP
		double F_G = NAN;
		double F_R = NAN;
		double rhs = NAN;
		double T_gas = NAN;
		double kappa = NAN;
		double fourPiB = NAN;
		// double dB_dTgas = NAN;
		// double dkappa_dTgas = NAN;
		double drhs_dEgas = NAN;
		double dFG_dEgas = NAN;
		double dFG_dErad = NAN;
		double dFR_dEgas = NAN;
		double dFR_dErad = NAN;
		double Src = NAN;
		double eta = NAN;
		double deltaErad = NAN;
		double deltaEgas = NAN;

		double Egas_guess = Egas0;
		double Erad_guess = Erad0;
		// const double T_floor = 1e-10;
		const double resid_tol = 1e-10;
		const int maxIter = 200;
		int n = 0;
		for (n = 0; n < maxIter; ++n) {

			// compute material temperature
			T_gas = RadSystem<problem_t>::ComputeTgasFromEgas(rho, Egas_guess);

			// compute opacity, emissivity
			kappa = RadSystem<problem_t>::ComputeOpacity(rho, T_gas);
			fourPiB = chat * a_rad * std::pow(T_gas, 4);

			// constant radiation energy source term
			// plus advection source term (for well-balanced/SDC integrators)
			Src = dt * ((chat * radEnergySource(i, j, k)) + advectionFluxes(i, j, k));

			// compute derivatives w/r/t T_gas
			const double dB_dTgas = (4.0 * fourPiB) / T_gas;
			const double dkappa_dTgas =
			    RadSystem<problem_t>::ComputeOpacityTempDerivative(rho, T_gas);

			// compute residuals
			rhs = dt * (rho * kappa) * (fourPiB - chat * Erad_guess);
			F_G = (Egas_guess - Egas0) + ((c / chat) * rhs);
			F_R = (Erad_guess - Erad0) - (rhs + Src);

			// check if converged
			if ((std::abs(F_G / Etot0) < resid_tol) &&
			    (std::abs(F_R / Etot0) < resid_tol)) {
				break;
			}

			// compute Jacobian elements
			const double c_v =
			    RadSystem<problem_t>::ComputeEgasTempDerivative(rho, T_gas);

			drhs_dEgas =
			    (rho * dt / c_v) *
			    (kappa * dB_dTgas + dkappa_dTgas * (fourPiB - chat * Erad_guess));

			dFG_dEgas = 1.0 + (c / chat) * drhs_dEgas;
			dFG_dErad = dt * (-(rho * kappa) * c);
			dFR_dEgas = -drhs_dEgas;
			dFR_dErad = 1.0 + dt * ((rho * kappa) * chat);

			// Update variables
			eta = -dFR_dEgas / dFG_dEgas;
			// eta = (eta > 0.0) ? eta : 0.0;

			deltaErad = -(F_R + eta * F_G) / (dFR_dErad + eta * dFG_dErad);
			deltaEgas = -(F_G + dFG_dErad * deltaErad) / dFG_dEgas;

			Egas_guess += deltaEgas;
			Erad_guess += deltaErad;

		} // END NEWTON-RAPHSON LOOP

		AMREX_ASSERT(std::abs(F_G / Etot0) < resid_tol); // NOLINT
		AMREX_ASSERT(std::abs(F_R / Etot0) < resid_tol); // NOLINT

		AMREX_ASSERT(Erad_guess > 0.0); // NOLINT
		AMREX_ASSERT(Egas_guess > 0.0); // NOLINT

		// store new radiation energy, gas energy
		consNew(i, j, k, radEnergy_index) = Erad_guess;
		consNew(i, j, k, gasEnergy_index) =
		    ComputeEgasFromEint(rho, x1GasMom0, 0, 0, Egas_guess); // modify in 3d

		// 2. Compute radiation flux update

		const double Frad_x_t0 = consPrev(i, j, k, x1RadFlux_index);
		const double Frad_x_t1 = (Frad_x_t0 + (dt * advectionFluxes(i, j, k))) /
					 (1.0 + (rho * kappa) * chat * dt);

		consNew(i, j, k, x1RadFlux_index) = Frad_x_t1;

		// 3. Compute conservative gas momentum update
		//	[N.B. should this step happen after the Lorentz	transform?]

		const double dF_x = Frad_x_t1 - Frad_x_t0;
		const double dx1Momentum = -dF_x / (c * chat);

		consNew(i, j, k, x1GasMomentum_index) += dx1Momentum;

		// 4. Update kinetic energy of gas

		const double dEkin = (vx0 * dx1Momentum); // modify in 3d
		consNew(i, j, k, gasEnergy_index) += dEkin;

		// Lorentz transform back to 'laboratory' frame
		// TransformIntoComovingFrame(-fluid_velocity);
	});
}

template <typename problem_t>
void RadSystem<problem_t>::ComputeSourceTermsExplicit(arrayconst_t &consPrev,
						      arrayconst_t &radEnergySource, array_t &src,
						      amrex::Box const &indexRange, amrex::Real dt)
{
	const double chat = c_hat_;
	const double a_rad = radiation_constant_;

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// load gas energy
		const auto rho = consPrev(i, j, k, gasDensity_index);
		const auto Egastot0 = consPrev(i, j, k, gasEnergy_index);
		const auto x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const auto Egas0 =
		    ComputeEintFromEgas(rho, x1GasMom0, 0, 0, Egastot0); // modify in 3d

		// load radiation energy, momentum
		const auto Erad0 = consPrev(i, j, k, radEnergy_index);
		const auto Frad0_x = consPrev(i, j, k, x1RadFlux_index);
		// compute material temperature
		const auto T_gas = RadSystem<problem_t>::ComputeTgasFromEgas(rho, Egas0);
		// compute opacity, emissivity
		const auto kappa = RadSystem<problem_t>::ComputeOpacity(rho, T_gas);
		const auto fourPiB = chat * a_rad * std::pow(T_gas, 4);
		// constant radiation energy source term
		const auto Src = dt * (chat * radEnergySource(i, j, k));
		// compute reaction term
		const auto rhs = dt * (rho * kappa) * (fourPiB - chat * Erad0);
		const auto Fx_rhs = -dt * chat * (rho * kappa) * Frad0_x;

		src(radEnergy_index, i) = rhs;
		src(x1RadFlux_index, i) = Fx_rhs;
	});
}

#endif // RADIATION_SYSTEM_HPP_
