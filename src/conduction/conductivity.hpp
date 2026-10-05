#ifndef CONDUCTIVITY_HPP_ // NOLINT
#define CONDUCTIVITY_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file conductivity.hpp
/// \brief Thermal conductivity models, selected by Physics_Traits<problem_t>::conduction_model and
///        Physics_Traits<problem_t>::conduction_geometry.
///
/// Conductivities are always specified as a full conductivity kappa (erg cm^-1 s^-1 K^-1), whether
/// through the conduction.* prefactors in the input file (ConductionModel::constant/spitzer) or through
/// a problem-specific computeConductivity (ConductionModel::problem_defined). The conduction solvers
/// only ever call quokka::conduction::EvaluateDiffusivity, which converts kappa to the diffusivity
/// chi = kappa / (n k_B) (cm^2 s^-1) of Sharma & Hammett (2007), so that q = -n k_B chi grad T.

#include <cmath>
#include <limits>

#include "AMReX.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "util/valarray.hpp"

/// \brief Returns the thermal conductivity at a point with density \p rho and temperature \p Tgas,
///        in units of erg cm^-1 s^-1 K^-1.
///
/// Specialize this in the problem file when using ConductionModel::problem_defined, e.g.
/// \code
/// template <> AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeConductivity<MyProblem>(amrex::Real rho, amrex::Real Tgas)
///     -> quokka::valarray<amrex::Real, 2>
/// {
/// 	return {kappa0 * std::pow(Tgas, 2.5), 0.0};
/// }
/// \endcode
///
/// \return {kappa_parallel, kappa_perp}. With ConductionGeometry::isotropic only kappa_parallel
///         (component 0) is used, as the isotropic conductivity. Both must be >= 0, and with
///         ConductionGeometry::anisotropic kappa_perp must be <= kappa_parallel: AnisoConduction limits the
///         normal flux term with the biased L2 limiter, which relies on (kappa_parallel - kappa_perp) * b_n^2 >= 0.
///         This is only checked in debug builds (AMREX_ASSERT in AnisoConduction).
template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeConductivity(amrex::Real /*rho*/, amrex::Real /*Tgas*/) -> quokka::valarray<amrex::Real, 2>
{
	static_assert(sizeof(problem_t) == 0, "computeConductivity must be specialized in the problem file when using ConductionModel::problem_defined");
	return {0.0, 0.0};
}

namespace quokka::conduction
{

/// \brief Conductivity prefactors read from the input file for the built-in conduction models.
///        Unused with ConductionModel::problem_defined. With ConductionGeometry::isotropic both are
///        set to conduction.conductivity_prefactor.
struct ConductivityParams {
	amrex::Real kappa0_par = 0.0;  // ConductionModel::constant: kappa (erg cm^-1 s^-1 K^-1);
				       // ConductionModel::spitzer: kappa0 in kappa = kappa0 * T^2.5 (erg cm^-1 s^-1 K^-3.5)
	amrex::Real kappa0_perp = 0.0; // as kappa0_par, across the magnetic field (ConductionGeometry::anisotropic only); must be <= kappa0_par
};

/// \brief Returns the thermal diffusivities {chi_parallel, chi_perp} = kappa / (n k_B) in cm^2 s^-1, with
///        n = rho / mean_molecular_weight, for the conductivity model and geometry set in Physics_Traits.
///        With ConductionGeometry::isotropic, chi_perp == chi_parallel. Returns zero with ConductionModel::none.
template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto EvaluateDiffusivity(amrex::Real rho, amrex::Real Tgas, ConductivityParams const &params)
    -> quokka::valarray<amrex::Real, 2>
{
	constexpr ConductionModel model = Physics_Traits<problem_t>::conduction_model;

	quokka::valarray<amrex::Real, 2> kappa{0.0, 0.0};
	if constexpr (model == ConductionModel::problem_defined) {
		kappa = computeConductivity<problem_t>(rho, Tgas);
	} else if constexpr (model == ConductionModel::spitzer) {
		const amrex::Real T_52 = std::pow(Tgas, 2.5);
		kappa = {params.kappa0_par * T_52, params.kappa0_perp * T_52};
	} else if constexpr (model == ConductionModel::constant) {
		kappa = {params.kappa0_par, params.kappa0_perp};
	} else { // ConductionModel::none
		amrex::ignore_unused(rho, Tgas, params);
		return kappa;
	}

	if constexpr (Physics_Traits<problem_t>::conduction_geometry == ConductionGeometry::isotropic) {
		kappa[1] = kappa[0];
	}

	constexpr amrex::Real mean_molecular_weight = quokka::EOS_Traits<problem_t>::mean_molecular_weight;
	if constexpr (model != ConductionModel::none) {
		// NaN is the only value that compares unequal to itself (std::isnan is not constexpr until C++23)
		static_assert(mean_molecular_weight == mean_molecular_weight, // NOLINT(misc-redundant-expression)
			      "Thermal conduction requires EOS_Traits::mean_molecular_weight to be set (chi = kappa / (n k_B), n = rho / mu).");
	}
	// floor rho so that a zero density gives a finite chi (and, after the solver multiplies by n, a zero kappa) rather than NaN
	const amrex::Real n = amrex::max(rho, std::numeric_limits<amrex::Real>::min()) / mean_molecular_weight;
	return kappa / (n * quokka::EOS<problem_t>::boltzmann_constant_);
}

} // namespace quokka::conduction

#endif // CONDUCTIVITY_HPP_
