#ifndef COSMOLOGY_HPP_
#define COSMOLOGY_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Cosmology.hpp
/// \brief Defines the Cosmology module for background evolution and source terms.

/// ## Physics background
///
/// ### Scale factor evolution
/// The cosmological scale factor a(t) evolves according to the Friedmann equation:
///   da/dt = a * H(a)
/// where H(a) is the Hubble parameter:
///   H(a) = H0 * E(a)
///   E(a) = sqrt( Omega_r/a^4 + Omega_m/a^3 + Omega_k/a^2 + Omega_L )
///
/// ### Comoving coordinate conventions
/// We work with COMOVING conserved variables stored in the state MultiFab:
///   rho_c   = rho_phys * a^3                (comoving mass density  -- constant in uniform expansion)
///   p_c     = rho_c * v_pec                 (comoving momentum      -- peculiar velocity)
///   eint_c  = eint_phys * a^3               (comoving internal energy density)
///   etot_c  = eint_c + |p_c|^2 / (2 rho_c) (comoving total energy density)
///
/// ### Source terms from the comoving Euler equations
/// 1. Momentum decay (Hubble drag):
///    d(p_c)/dt = -H * p_c
///    Exact solution: p_c(t) = p_c(0) * (a_0 / a(t))
///
/// 2. Expansion cooling (adiabatic):
///    For a gamma-law gas: P ∝ rho_phys^gamma ∝ a^(-3*gamma)
///    eint_phys = P/(gamma-1) ∝ a^(-3*gamma)
///    eint_c = eint_phys * a^3 ∝ a^(-3*(gamma-1))
///    d(eint_c)/dt = -3*(gamma - 1) * H * eint_c
///    Exact solution: eint_c(t) = eint_c(0) * (a_0 / a(t))^[3*(gamma - 1)]
///
/// For gamma=5/3 (monatomic ideal gas): T ∝ a^(-2), eint_c ∝ a^(-2).

#include <cmath>

#include "AMReX.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "math/ODEIntegrate.hpp"  

namespace quokka::cosmology
{

/// @brief Parameters for the cosmological model (LCDM by default)
struct CosmologyParams {
	amrex::Real H0{C::Hubble_const};      ///< Hubble constant at z=0: 32.407764868e-19 s^-1, from quokka//extern/Microphysics/constants/fundamental_constants.H
	amrex::Real Omega_m{0.30966};	      ///< Matter density parameter
	amrex::Real Omega_r{9.13896e-05};	  ///< Radiation density parameter
	amrex::Real Omega_L{0.68885};	      ///< Dark energy (Lambda) density parameter
	// Omega_k = 1 - (Omega_m + Omega_r + Omega_L)  [derived]

	amrex::Real Omega_b{0.04897};         ///< Ordinary matter density parameter
	amrex::Real Omega_dm{0.26069};	      ///< DM density parameter
	
	void validate() const {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(Omega_b + Omega_dm - Omega_m) < 1e-6, 
    	"Error: The sum of Omega_b and Omega_dm must equal Omega_m!");
	}
};

/// @brief Compute the dimensionless Hubble factor E(a) = H(a)/H0
///
/// \param a   Scale factor (a=1 today, a<1 in the past)
/// \param cosmo Cosmology parameters
/// \return E(a) = sqrt(Omega_r/a^4 + Omega_m/a^3 + Omega_k/a^2 + Omega_L)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto HubbleFactor(amrex::Real a, CosmologyParams const &cosmo) -> amrex::Real
{
	const amrex::Real Omega_k = 1.0 - cosmo.Omega_r - cosmo.Omega_m - cosmo.Omega_L;
	const amrex::Real a2 = a * a;
	const amrex::Real a3 = a2 * a;
	const amrex::Real a4 = a2 * a2;

	// E^2(a) = sum of energy density contributions, each diluted by expansion
	const amrex::Real E2 = cosmo.Omega_r / a4 + cosmo.Omega_m / a3 + Omega_k / a2 + cosmo.Omega_L;
	// std::max guards against sqrt of a tiny negative number from floating-point round-off
	return std::sqrt(std::max(E2, static_cast<amrex::Real>(0.0)));
}


/// @brief Evolve the scale factor a(t) from a_old to a_old+dt using an adaptive (according to the
/// relative error) RK12 (Heun and Euler)
///
/// \param a_old        Starting scale factor
/// \param dt           Time interval to integrate over
/// \param cosmo        Cosmology parameters
/// \return             New scale factor a(t+dt)

// Definition of the functor for the ODE
struct FriedmannRhsFunctor {
	// Data member for the ODE
	CosmologyParams const &cosmo;

	// Constructor (explicit to prevent accidental data coversion)
	AMREX_GPU_HOST_DEVICE explicit FriedmannRhsFunctor(CosmologyParams const &cosmo_in) : cosmo(cosmo_in) {
	}  

	// Functor for the Friedmann rhs: overloading of the () operator:
	// t is the current time
	// y_data is the input current status
	// y_rhs is the output, the derivative dy/dt
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
	auto operator()(amrex::Real /* t */, quokka::valarray<amrex::Real, 1> const &y_data, quokka::valarray<Real, 1> &y_rhs) const -> int {
		const amrex::Real a = y_data[0];                   // initial scale factor value from the state vector
		y_rhs[0] = a * cosmo.H0 * HubbleFactor(a, cosmo);  // da/dt = a * H(a)
		return 0;
	}
};

/// [[nodiscard]]: the compiler warns if the caller discards the return value.
// Evolve the scale factor solving the Friedmann equation
[[nodiscard]] inline auto evolveScaleFactor(amrex::Real a_old, amrex::Real dt, CosmologyParams const &cosmo) -> amrex::Real {

	// Check the scale factor is not negative
	AMREX_ASSERT_WITH_MESSAGE(a_old > 0.0, "Scale factor a_old must be positive!");

	// Inputs for the integrator
	FriedmannRhsFunctor rhs(cosmo);
	quokka::valarray<amrex::Real, 1> y ={a_old};
    quokka::valarray<amrex::Real, 1> abstol = {1.0e-12 * a_old}; // absolute tolerance
    const Real rtol = 1.0e-8;                                    // relative tolerance

	int steps_taken = 0;   // counter of the substeps to cover the integration dt
	rk_adaptive_integrate(rhs, 0, y, dt, rtol, abstol, steps_taken);

	return y[0];           // new scale factor a(t+dt)
}



/// @brief Apply cosmological source terms (Hubble drag + expansion cooling) over [a_old, a_new]
///
/// Uses the EXACT analytic solution for each source term, so accuracy is limited only by
/// how well a_old and a_new represent the true scale factors at the start and end of the step,
/// not by the size of the timestep dt. This makes the source term update unconditionally stable.
///
/// Called as part of Strang splitting from addStrangSplitSourcesWithBuiltin().
/// By the time this is called, 'dt' passed from the splitting framework is already 0.5*dt_lev.
///
/// \param state  Cell-centred conserved state MultiFab (modified in place)
/// \param a_old  Scale factor at the beginning of this sub-step
/// \param a_new  Scale factor at the end of this sub-step
template <typename problem_t> void applyCosmologicalSourceTerms(amrex::MultiFab &state, amrex::Real const a_old, amrex::Real const a_new)
{
	// Ratio of scale factors
	const amrex::Real ratio = a_old / a_new;

	// Momentum scales as p_c ∝ 1/a  -> p_new = p_old * (a_old/a_new)
	const amrex::Real mom_ratio = ratio;

	// Internal energy scales as e_c ∝ a^{-3(gamma-1)}  ->  e_new = e_old * ratio^{3(gamma-1)}
	// Here gamma is read on the CPU because static constexpr members of template classes
	// can be ill-formed inside AMREX_GPU_DEVICE lambdas under some NVCC versions.
	const amrex::Real gamma = quokka::EOS_Traits<problem_t>::gamma;
	const amrex::Real eint_ratio = std::pow(ratio, 3.0 * (gamma - 1.0));

	// Read variable indices once on the CPU and capture by value into the GPU lambda
	const int density_idx = HydroSystem<problem_t>::density_index;
	const int px_idx      = HydroSystem<problem_t>::x1Momentum_index;
	const int py_idx      = HydroSystem<problem_t>::x2Momentum_index;
	const int pz_idx      = HydroSystem<problem_t>::x3Momentum_index;
	const int etot_idx    = HydroSystem<problem_t>::energy_index;
	const int eint_idx    = HydroSystem<problem_t>::internalEnergy_index;

	// state.arrays() returns a MultiArray4 proxy that covers all boxes on this
	// MPI rank simultaneously, enabling a single batch kernel launch across all boxes
	auto const state_arrs = state.arrays();

	amrex::ParallelFor(state, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real rho = state_arrs[bx](i, j, k, density_idx);
		// Skip cells at or below the density floor (rho<=0 would cause division by zero in KE)
		if (rho <= 0.0) {
			return;
		}

		// --- 1. Hubble drag: scale each momentum component by a_old/a_new ---
		const amrex::Real px_new = state_arrs[bx](i, j, k, px_idx) * mom_ratio;
		const amrex::Real py_new = state_arrs[bx](i, j, k, py_idx) * mom_ratio;
		const amrex::Real pz_new = state_arrs[bx](i, j, k, pz_idx) * mom_ratio;

		// --- 2. Expansion cooling: scale internal energy by (a_old/a_new)^{3(gamma-1)} ---
		const amrex::Real eint_new = state_arrs[bx](i, j, k, eint_idx) * eint_ratio;

		// --- 3. Recompute total energy from updated momentum and internal energy ---
		// We do NOT scale etot directly because momentum and eint scale differently.
		// Recomputing from scratch avoids accumulated floating-point inconsistency
		// between etot and eint that would confuse the dual-energy formalism.
		const amrex::Real KE_new = 0.5 * (px_new * px_new + py_new * py_new + pz_new * pz_new) / rho;
		const amrex::Real etot_new = eint_new + KE_new;

		// Write back all updated quantities
		state_arrs[bx](i, j, k, px_idx)   = px_new;
		state_arrs[bx](i, j, k, py_idx)   = py_new;
		state_arrs[bx](i, j, k, pz_idx)   = pz_new;
		state_arrs[bx](i, j, k, eint_idx) = eint_new;
		state_arrs[bx](i, j, k, etot_idx) = etot_new;
	});
	amrex::Gpu::streamSynchronize(); // wait for the GPU kernel to finish before returning

}

/// @brief Perform one half-step of the Strang-split cosmological source term update
///
/// This is the entry point called from QuokkaSimulation::addStrangSplitSourcesWithBuiltin().
/// The Strang splitting scheme ensures second-order accuracy by calling this function
/// twice per full timestep (once with 0.5*dt_lev before the hydro advance, once after).
/// Each call evolves a_now_ by dt_half and applies the corresponding analytic source terms.
///
/// \param state    Cell-centred conserved state MultiFab (modified in place)
/// \param a_begin  Scale factor at the start of this half-step
/// \param dt_half  Duration of the half-step (= 0.5 * dt_lev, provided by Strang splitting)
/// \param cosmo    Cosmology parameters
/// \return         New scale factor at the end of the half-step
template <typename problem_t>
auto applyCosmologyHalfStep(amrex::MultiFab &state, amrex::Real a_begin, amrex::Real dt_half, CosmologyParams const &cosmo) -> amrex::Real
{
	const amrex::Real a_end = evolveScaleFactor(a_begin, dt_half, cosmo);
	applyCosmologicalSourceTerms<problem_t>(state, a_begin, a_end);
	return a_end;
}

/// @brief Print cosmological parameters to standard output (MPI rank 0 only)
///
/// The conversion from H0 [s^-1] to [km/s/Mpc] uses:
///   H0 [km/s/Mpc] = H0 [s^-1] * (Mpc in cm) / (km in cm)
///                 = H0 * (parsec [cm] * 1e6) / 1e5
inline void printCosmologyInfo(CosmologyParams const &cosmo, amrex::Real a_now)
{
	const amrex::Real H0_km_s_Mpc = cosmo.H0 * (C::parsec * 1.0e6) / 1.0e5;
	const amrex::Real Omega_k = 1.0 - cosmo.Omega_r - cosmo.Omega_m - cosmo.Omega_L;
	amrex::Print() << "\nCosmological parameters:\n";
	amrex::Print() << "  H0      = " << H0_km_s_Mpc << " km/s/Mpc\n";
	amrex::Print() << "  Omega_m = " << cosmo.Omega_m << "\n";
	amrex::Print() << "  Omega_b = " << cosmo.Omega_b << "\n";
	amrex::Print() << "  Omega_dm = " << cosmo.Omega_dm << "\n";
	amrex::Print() << "  Omega_r = " << cosmo.Omega_r << "\n";
	amrex::Print() << "  Omega_L = " << cosmo.Omega_L << "\n";
	amrex::Print() << "  Omega_k = " << Omega_k << " (derived, 0 = flat)\n";
	amrex::Print() << "  a_init  = " << a_now << " (z_init = " << (1.0 / a_now - 1.0) << ")\n\n";
}

} // namespace quokka::cosmology

#endif // COSMOLOGY_HPP_