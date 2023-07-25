#ifndef EOS_HPP_
#define EOS_HPP_
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file EOS.hpp
/// \brief A class for equation of state calculations.

#include <cmath>
#include <optional>

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "physics_info.hpp"
#include <AMReX_Print.H>

#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"

#ifdef PRIMORDIAL_CHEM
#include "actual_eos_data.H"
#endif

namespace quokka
{

// specify default values for ideal gamma-law EOS
//
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5. / 3.;     // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	static constexpr double mean_molecular_weight = NAN;
	static constexpr double boltzmann_constant = C::k_B;
};

template <typename problem_t> class EOS
{

      public:
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = {}) -> amrex::Real;
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = {}) -> amrex::Real;
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(amrex::Real rho, amrex::Real Tgas, const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars = {})
	    -> amrex::Real;

      private:
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real boltzmann_constant_ = EOS_Traits<problem_t>::boltzmann_constant;
	static constexpr amrex::Real mean_molecular_weight_ = EOS_Traits<problem_t>::mean_molecular_weight;
};

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars)
    -> amrex::Real
{
	// return temperature for an ideal gas
	amrex::Real Tgas = NAN;

#ifdef PRIMORDIAL_CHEM
	burn_t chemstate;
	chemstate.rho = rho;
	chemstate.e = Eint / rho;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_re, chemstate);
	Tgas = chemstate.T;
#else
	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.e = Eint / rho;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_re, estate);
		// scale returned temperature in case boltzmann constant is dimensionless
		Tgas = estate.T * C::k_B / boltzmann_constant_;
	}
#endif
	return Tgas;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars)
    -> amrex::Real
{
	// return internal energy density for a gamma-law ideal gas
	amrex::Real Eint = NAN;

#ifdef PRIMORDIAL_CHEM
	burn_t chemstate;
	chemstate.rho = rho;
	// Define and initialize Tgas here
	amrex::Real const Tgas_value = Tgas;
	chemstate.T = Tgas_value;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_rt, chemstate);
	Eint = chemstate.e * chemstate.rho;
#else
	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.T = Tgas;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_rt, estate);
		Eint = estate.e * rho * boltzmann_constant_ / C::k_B;
	}
#endif
	return Eint;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto
EOS<problem_t>::ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
					  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> massScalars) -> amrex::Real
{
	// compute derivative of internal energy w/r/t temperature
	amrex::Real dEint_dT = NAN;

#ifdef PRIMORDIAL_CHEM
	burn_t chemstate;
	chemstate.rho = rho;
	// we don't need Tgas to find chemstate.dedT, but we still need to initialize chemstate.T because we are using the 'rt' EOS mode
	chemstate.T = NAN;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_rt, chemstate);
	dEint_dT = chemstate.dedT * chemstate.rho;
#else
	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.T = Tgas;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_rt, estate);
		dEint_dT = estate.dedT * rho * boltzmann_constant_ / C::k_B;
	}
#endif
	return dEint_dT;
}

} // namespace quokka

#endif // EOS_HPP_
