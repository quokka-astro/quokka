#ifndef EOS_HPP_
#define EOS_HPP_
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file EOS.hpp
/// \brief A class for equation of state calculations.

#include <cmath>
#include <string>

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include <fundamental_constants.H>

namespace quokka
{
static constexpr double hydrogen_mass_cgs = C::m_p + C::m_e; // cgs

// specify default values for ideal gamma-law EOS
//
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5.0 / 3.0;   // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	static constexpr double mean_molecular_weight = NAN;
};

template <typename problem_t> class EOS
{
      public:
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint) -> amrex::Real;
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas) -> amrex::Real;
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeEintTempDerivative(amrex::Real rho, amrex::Real Tgas) -> amrex::Real;

      private:
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real mean_molecular_weight_ = EOS_Traits<problem_t>::mean_molecular_weight;
};

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint) -> amrex::Real
{
	// return temperature for an ideal gas

	burn_t chemstate;
	chemstate.rho = rho;
	chemstate.e = Eint / rho;
	chemstate.mu = mean_molecular_weight_;
	eos(eos_input_re,
	    chemstate); // this will cause an error when primordial chem is run with hydro, because we also need to input values of the mass scalars in
			// chemstate.xn
	amrex::Real Tgas = chemstate.T;

	return Tgas;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas) -> amrex::Real
{
	// return internal energy density for a gamma-law ideal gas

	burn_t chemstate;
	chemstate.rho = rho;
	chemstate.T = Tgas;
	chemstate.mu = mean_molecular_weight_;
	eos(eos_input_rt,
	    chemstate); // this will cause an error when primordial chem is run with hydro, because we also need to input values of the mass scalars in
			// chemstate.xn
	amrex::Real const Eint = chemstate.e * chemstate.rho;

	return Eint;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas) -> amrex::Real
{
	// compute derivative of internal energy w/r/t temperature

	burn_t chemstate;
	chemstate.rho = rho;
	chemstate.T = Tgas;
	chemstate.mu = mean_molecular_weight_;
	eos(eos_input_rt,
	    chemstate); // this will cause an error when primordial chem is run with hydro, because we also need to input values of the mass scalars in
			// chemstate.xn
	amrex::Real const dEint_dT = chemstate.dedT * chemstate.rho;

	return dEint_dT;
}

} // namespace quokka

#endif // EOS_HPP_
