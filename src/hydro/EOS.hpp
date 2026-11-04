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
#include <type_traits>

#include "util/Optional.hpp"

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "physics_info.hpp"
#include <AMReX_Print.H>

#include "cooling/EOSTabulatedRegistry.hpp"
#ifdef CHEMISTRY
#include "eos.H"
#endif
#ifdef PHOTOCHEMISTRY
#include "network.hpp"
#include "photochemistry/PhotochemEOS.hpp"
#endif

#ifdef CHEMISTRY
#include "actual_eos_data.H"
#endif

namespace quokka
{

// forward declarations of EOS backends
template <typename problem_t> struct EOSIdeal;
#ifdef CHEMISTRY
template <typename problem_t> struct EOSMicrophysics;
#endif
#ifdef PHOTOCHEMISTRY
template <typename problem_t> struct EOSPhotochem;
#endif
template <typename problem_t> struct EOSTabulated;

// Single source of truth for the default EOS backend selection, forward-declared here
// so EOS_Traits can reference it. Fully defined after EOSMicrophysics below.
template <typename problem_t> struct DefaultEOSBackend;

// Primary EOS_Traits template. Provides default values for ideal gamma-law EOS
// and selects the compile-time EOS backend. Full specializations need only define
// the trait values they override; EOSBackend defaults via SFINAE if omitted.
//
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5. / 3.;     // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	static constexpr double mean_molecular_weight = NAN;

	using EOSBackend = typename DefaultEOSBackend<problem_t>::type;
};

// ==================== EOSIdeal backend ====================
// gamma-law ideal gas (the current #else path). Always available.

template <typename problem_t> struct EOSIdeal {
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = false;
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real mean_molecular_weight_ = EOS_Traits<problem_t>::mean_molecular_weight;
	static constexpr amrex::Real boltzmann_constant_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return C::k_B;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::boltzmann_constant;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			return C::k_B /
			       (Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_mass /
				(Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time) / Physics_Traits<problem_t>::unit_temperature);
		}
	}();

	// Gamma-law state in the form the Microphysics gamma_law EOS used: temperature in kelvin with C::k_B, specific internal energy e.
	struct GammaLawState {
		amrex::Real rho;
		amrex::Real T;
		amrex::Real p;
		amrex::Real e;
	};

	// Input floors of the Microphysics eos() wrapper, as set by eos_init(small_temp = 1e-10, small_dens = 1e-100).
	static constexpr amrex::Real eos_min_temp_ = 1.0e-10;
	static constexpr amrex::Real eos_min_dens_ = 1.0e-100;
	static constexpr amrex::Real eos_min_e_ = 1.0e-200;
	static constexpr amrex::Real eos_min_p_ = 1.0e-200;
	static constexpr amrex::Real eos_max_ = 1.0e200;

	// p and e from rho and T, without input resets
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto GammaLawFromValidRhoT(amrex::Real rho, amrex::Real T) -> GammaLawState
	{
		const amrex::Real p = rho * C::k_B * T / mean_molecular_weight_;
		const amrex::Real e = p / ((gamma_ - 1.0) * rho);
		return GammaLawState{.rho = rho, .T = T, .p = p, .e = e};
	}

	// Clamp by value: a NaN x gives lo, as Microphysics's min(hi, max(lo, x)) did. Taking the bounds by value also avoids
	// binding references to the static constexpr members in device code.
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ClampToRange(amrex::Real x, amrex::Real lo, amrex::Real hi) -> amrex::Real
	{
		return std::fmin(std::fmax(x, lo), hi);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto GammaLawFromRhoT(amrex::Real rho, amrex::Real T) -> GammaLawState
	{
		return GammaLawFromValidRhoT(ClampToRange(rho, eos_min_dens_, eos_max_), ClampToRange(T, eos_min_temp_, eos_max_));
	}

	// An out-of-range e is replaced by the state at the minimum temperature, as Microphysics's eos_reset() did
	// (its input state had T = 0, which it raised to the floor).
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto GammaLawFromRhoE(amrex::Real rho, amrex::Real e) -> GammaLawState
	{
		if (e < eos_min_e_ || e > eos_max_) {
			return GammaLawFromRhoT(rho, 0.0);
		}
		const amrex::Real T = e * mean_molecular_weight_ * (gamma_ - 1.0) / C::k_B;
		return GammaLawFromValidRhoT(ClampToRange(rho, eos_min_dens_, eos_max_), T);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto GammaLawFromRhoP(amrex::Real rho, amrex::Real p) -> GammaLawState
	{
		if (p < eos_min_p_ || p > eos_max_) {
			return GammaLawFromRhoT(rho, 0.0);
		}
		const amrex::Real rho_clamped = ClampToRange(rho, eos_min_dens_, eos_max_);
		const amrex::Real T = p * mean_molecular_weight_ / (C::k_B * rho_clamped);
		return GammaLawFromValidRhoT(rho_clamped, T);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real Tgas = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			Tgas = GammaLawFromRhoE(rho, Eint / rho).T * C::k_B / boltzmann_constant_;
		}
		return Tgas;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real Eint = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			Eint = GammaLawFromRhoT(rho, Tgas).e * rho * boltzmann_constant_ / C::k_B;
		}
		return Eint;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real Eint = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			Eint = GammaLawFromRhoP(rho, Pressure).e * rho;
		}
		return Eint;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		amrex::Real dEint_dT = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			const GammaLawState s = GammaLawFromRhoT(rho, Tgas);
			dEint_dT = (s.e / s.T) * rho * boltzmann_constant_ / C::k_B;
		}
		return dEint_dT;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		amrex::Real deint_dRho = NAN;
		amrex::Real deint_dP = NAN;
		amrex::Real dRho_dP = NAN;
		amrex::Real dP_dRho_s = NAN;
		amrex::Real G = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			const GammaLawState s = GammaLawFromRhoP(rho, P);
			deint_dRho = 0.0;
			deint_dP = 1.0 / ((s.p / s.T) / (s.e / s.T));
			dRho_dP = 1.0 / ((s.p / s.rho) * C::k_B / boltzmann_constant_);
			dP_dRho_s = gamma_ * s.p / s.rho;
			G = 0.5 * (1.0 + gamma_);
		}
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		amrex::Real P = NAN;

		if constexpr (gamma_ == 1.0) {
			static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
			amrex::ignore_unused(Eint);
			amrex::ignore_unused(massScalars);
			P = rho * EOS_Traits<problem_t>::cs_isothermal * EOS_Traits<problem_t>::cs_isothermal;
			return P;
		}
		amrex::ignore_unused(massScalars);

		const amrex::Real e = (rho == 0.0) ? 0.0 : Eint / rho;
		P = GammaLawFromRhoE(rho, e).p;
		return P;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real cs = NAN;

		if constexpr (gamma_ == 1.0) {
			static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
			amrex::ignore_unused(rho);
			amrex::ignore_unused(Pressure);
			amrex::ignore_unused(massScalars);
			cs = EOS_Traits<problem_t>::cs_isothermal;
			return cs;
		}
		amrex::ignore_unused(massScalars);

		const GammaLawState s = GammaLawFromRhoP(rho, Pressure);
		cs = std::sqrt(gamma_ * s.p / s.rho);
		return cs;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		amrex::Real cs = NAN;

		if constexpr (gamma_ == 1.0) {
			static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
			amrex::ignore_unused(rho);
			amrex::ignore_unused(Pressure);
			cs = EOS_Traits<problem_t>::cs_isothermal;
		} else {
			cs = std::sqrt(Pressure / rho);
		}
		return cs;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real /*rho*/, amrex::Real /*Eint*/,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/ = {}) -> amrex::Real
	{
		// sizeof(problem_t)==0: fires only on instantiation, not at parse time (C++20 ill-formed NDR workaround)
		static_assert(sizeof(problem_t) == 0, "ComputeEntropyFromRhoEint is only supported by the EOSTabulated backend");
		return 0.0;
	}
};

// ==================== EOSMicrophysics backend ====================
// Only compiled when CHEMISTRY is defined.

#ifdef CHEMISTRY
template <typename problem_t> struct EOSMicrophysics {
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = false;
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real boltzmann_constant_ = EOSIdeal<problem_t>::boltzmann_constant_;

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.e = Eint / rho;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_re, chemstate);
		return chemstate.T;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.T = Tgas;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rt, chemstate);
		return chemstate.e * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.p = Pressure;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rp, chemstate);
		return chemstate.e * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.T = Tgas;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rt, chemstate);
		return chemstate.dedT * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.p = P;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rp, chemstate);
		const amrex::Real deint_dRho = chemstate.dedr;
		const amrex::Real deint_dP = 1.0 / chemstate.dpde;
		amrex::Real dRho_dP = 1.0 / (chemstate.dpdr * C::k_B / boltzmann_constant_);
		const amrex::Real dP_dRho_s = chemstate.cs * chemstate.cs;
		const amrex::Real G = chemstate.G;
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		static_assert(gamma_ != 1.0, "EOSMicrophysics does not support isothermal (gamma=1) problems.");
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.e = Eint / rho;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_re, chemstate);
		return chemstate.p;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		static_assert(gamma_ != 1.0, "EOSMicrophysics does not support isothermal (gamma=1) problems.");
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.p = Pressure;
		for (double &ii : chemstate.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rp, chemstate);
		return chemstate.cs;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		static_assert(gamma_ == 1.0, "ComputeIsothermalSoundSpeed does not support general EOS");
		static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
		amrex::ignore_unused(rho);
		amrex::ignore_unused(Pressure);
		return EOS_Traits<problem_t>::cs_isothermal;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real /*rho*/, amrex::Real /*Eint*/,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/ = {}) -> amrex::Real
	{
		// sizeof(problem_t)==0: fires only on instantiation, not at parse time (C++20 ill-formed NDR workaround)
		static_assert(sizeof(problem_t) == 0, "ComputeEntropyFromRhoEint is only supported by the EOSTabulated backend");
		return 0.0;
	}
};
#endif // CHEMISTRY

// ==================== EOSPhotochem backend ====================
// Multi-gamma ideal gas of the photochemistry network's species (src/photochemistry/PhotochemEOS.hpp), with the input
// floors of the Microphysics eos() wrapper it replaces (minimum temperature 1e-10 K, minimum density 1e-100, and a reset
// to the minimum temperature when e or p is out of range). Only compiled when PHOTOCHEMISTRY is defined.

#ifdef PHOTOCHEMISTRY
template <typename problem_t> struct EOSPhotochem {
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = false;
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real boltzmann_constant_ = EOSIdeal<problem_t>::boltzmann_constant_;

	static constexpr amrex::Real eos_min_temp_ = 1.0e-10;
	static constexpr amrex::Real eos_min_dens_ = 1.0e-100;
	static constexpr amrex::Real eos_min_ep_ = 1.0e-200; // minimum specific energy and pressure
	static constexpr amrex::Real eos_max_ = 1.0e200;

	struct State {
		amrex::Real rho;   // clamped density
		amrex::Real T;	   // temperature [K]
		amrex::Real e;	   // specific internal energy
		amrex::Real dedT;  // de/dT
		amrex::Real f_avg; // sum_i n_i/(gamma_i - 1) / sum_i n_i: heat capacity per particle in units of k_B
	};

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ClampToRange(amrex::Real x, amrex::Real lo, amrex::Real hi) -> amrex::Real
	{
		return std::fmin(std::fmax(x, lo), hi);
	}

	// Number densities from the mass scalars. Without them, Microphysics used xn = -1 for every species; T(e) depends
	// only on ratios of the n_i, so equal number densities give the same results.
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	NumberDensities(quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		amrex::GpuArray<amrex::Real, NumSpec> n{};
		for (int i = 0; i < NumSpec; ++i) {
			n[i] = massScalars ? (*massScalars)[i] / spmasses[i] : 1.0;
		}
		return n;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto HeatCapacityPerParticle(amrex::GpuArray<amrex::Real, NumSpec> const &n) -> amrex::Real
	{
		amrex::Real sum_n = 0.0;
		amrex::Real sum_f = 0.0;
		for (int i = 0; i < NumSpec; ++i) {
			sum_n += n[i];
			sum_f += n[i] / (gammas[i] - 1.0);
		}
		return sum_f / sum_n;
	}

	// (rho, T) input: rho and T are clamped to the floors
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto FromRhoT(amrex::Real rho, amrex::Real T, amrex::GpuArray<amrex::Real, NumSpec> const &n)
	    -> State
	{
		State s{};
		s.rho = ClampToRange(rho, eos_min_dens_, eos_max_);
		s.T = ClampToRange(T, eos_min_temp_, eos_max_);
		s.e = quokka::photochemistry::EnergyFromTemperature<NumSpec>(s.T, n, spmasses, gammas);
		s.dedT = quokka::photochemistry::dEnergy_dT<NumSpec>(n, spmasses, gammas);
		s.f_avg = HeatCapacityPerParticle(n);
		return s;
	}

	// (rho, e) input: an out-of-range e is replaced by the state at the minimum temperature; T itself is not floored
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto FromRhoE(amrex::Real rho, amrex::Real e, amrex::GpuArray<amrex::Real, NumSpec> const &n)
	    -> State
	{
		if (e < eos_min_ep_ || e > eos_max_) {
			return FromRhoT(rho, 0.0, n);
		}
		State s{};
		s.rho = ClampToRange(rho, eos_min_dens_, eos_max_);
		s.e = e;
		s.T = quokka::photochemistry::TemperatureFromEnergy<NumSpec>(e, n, spmasses, gammas);
		s.dedT = quokka::photochemistry::dEnergy_dT<NumSpec>(n, spmasses, gammas);
		s.f_avg = HeatCapacityPerParticle(n);
		return s;
	}

	// (rho, p) input: an out-of-range p is replaced by the state at the minimum temperature
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto FromRhoP(amrex::Real rho, amrex::Real p, amrex::GpuArray<amrex::Real, NumSpec> const &n)
	    -> State
	{
		if (p < eos_min_ep_ || p > eos_max_) {
			return FromRhoT(rho, 0.0, n);
		}
		State s{};
		s.rho = ClampToRange(rho, eos_min_dens_, eos_max_);
		s.f_avg = HeatCapacityPerParticle(n);
		s.e = p * s.f_avg / s.rho;
		s.T = quokka::photochemistry::TemperatureFromEnergy<NumSpec>(s.e, n, spmasses, gammas);
		s.dedT = quokka::photochemistry::dEnergy_dT<NumSpec>(n, spmasses, gammas);
		return s;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto PressureOf(State const &s) -> amrex::Real { return s.rho * s.e / s.f_avg; }

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto SoundSpeedOf(State const &s) -> amrex::Real
	{
		return std::sqrt((1.0 + 1.0 / s.f_avg) * PressureOf(s) / s.rho);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return FromRhoE(rho, Eint / rho, NumberDensities(massScalars)).T;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		const State s = FromRhoT(rho, Tgas, NumberDensities(massScalars));
		return s.e * s.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		const State s = FromRhoP(rho, Pressure, NumberDensities(massScalars));
		return s.e * s.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		const State s = FromRhoT(rho, Tgas, NumberDensities(massScalars));
		return s.dedT * s.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		const State s = FromRhoP(rho, P, NumberDensities(massScalars));
		const amrex::Real p = PressureOf(s);
		const amrex::Real dpdT = p / s.T;
		const amrex::Real dpdr = p / s.rho;
		const amrex::Real deint_dRho = 0.0;
		const amrex::Real deint_dP = 1.0 / (dpdT / s.dedT);
		const amrex::Real dRho_dP = 1.0 / (dpdr * C::k_B / boltzmann_constant_);
		const amrex::Real cs = SoundSpeedOf(s);
		const amrex::Real dP_dRho_s = cs * cs;
		const amrex::Real G = 0.5 * (1.0 + (1.0 + 1.0 / s.f_avg));
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		static_assert(gamma_ != 1.0, "EOSPhotochem does not support isothermal (gamma=1) problems.");
		return PressureOf(FromRhoE(rho, Eint / rho, NumberDensities(massScalars)));
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		static_assert(gamma_ != 1.0, "EOSPhotochem does not support isothermal (gamma=1) problems.");
		return SoundSpeedOf(FromRhoP(rho, Pressure, NumberDensities(massScalars)));
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		static_assert(gamma_ == 1.0, "ComputeIsothermalSoundSpeed does not support general EOS");
		static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
		amrex::ignore_unused(rho);
		amrex::ignore_unused(Pressure);
		return EOS_Traits<problem_t>::cs_isothermal;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real /*rho*/, amrex::Real /*Eint*/,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/ = {}) -> amrex::Real
	{
		// sizeof(problem_t)==0: fires only on instantiation, not at parse time (C++20 ill-formed NDR workaround)
		static_assert(sizeof(problem_t) == 0, "ComputeEntropyFromRhoEint is only supported by the EOSTabulated backend");
		return 0.0;
	}
};
#endif // PHOTOCHEMISTRY

// ==================== EOSTabulated backend ====================
// Temperature methods read the resampled table; all other methods delegate to EOSIdeal.

template <typename problem_t> struct EOSTabulated {
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = true;
	static constexpr amrex::Real gamma_ = EOSIdeal<problem_t>::gamma_;
	static constexpr amrex::Real boltzmann_constant_ = EOSIdeal<problem_t>::boltzmann_constant_;

	// Temperature methods — use the resampled table via the global registry
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		return ResampledCooling::ComputeTgasFromEgas(rho, Eint, get_tables());
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		return ResampledCooling::ComputeEgasFromTgas(rho, Tgas, get_tables());
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		// One root-find, then use DataTable::partial_derivative for ∂T/∂eint_specific.
		// Table axes: (rho, eint_specific = Eint/rho). dEint_density/dT = rho / (∂T/∂eint_specific).
		const amrex::Real Eint = ComputeEintFromTgas(rho, Tgas, massScalars);
		auto const &tables = get_tables();
		const amrex::Real dT_deint = tables.all_tables.partial_derivative({rho, Eint / rho}, 1, ResampledCooling::TEMPERATURE_IDX);
		AMREX_ASSERT(dT_deint > amrex::Real(0.0));
		return rho / dT_deint;
	}

	// Non-temperature methods — delegate to EOSIdeal
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return EOSIdeal<problem_t>::ComputeEintFromPres(rho, Pressure, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		return EOSIdeal<problem_t>::ComputeOtherDerivatives(rho, P, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		return EOSIdeal<problem_t>::ComputePressure(rho, Eint, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return EOSIdeal<problem_t>::ComputeSoundSpeed(rho, Pressure, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		return EOSIdeal<problem_t>::ComputeIsothermalSoundSpeed(rho, Pressure);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		return ResampledCooling::ComputeEntropyFromRhoEint(rho, Eint, get_tables());
	}

      private:
	// Returns the device or host table handle appropriate for the current execution context.
	// The registration invariant (non-null pointer) is checked once at setup; only a
	// debug-mode assert is needed here to avoid per-cell overhead in Release GPU kernels.
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto get_tables() -> ResampledCooling::resampledGpuConstTables const &
	{
		auto *reg = ResampledCooling::getEOSTabulatedRegistry();
		AMREX_ASSERT(reg != nullptr);
		AMREX_IF_ON_DEVICE((return reg->device;))
		AMREX_IF_ON_HOST((return reg->host;))
	}
};

// DefaultEOSBackend — definition (forward-declared near EOS_Traits above).
// Both EOS_Traits (primary template) and EOSBackendHelper (SFINAE fallback) use this
// so the default-backend policy lives in exactly one place.
template <typename T> struct DefaultEOSBackend {
#if defined(CHEMISTRY)
	using type = EOSMicrophysics<T>;
#elif defined(PHOTOCHEMISTRY)
	using type = EOSPhotochem<T>;
#else
	using type = EOSIdeal<T>;
#endif
};

// ==================== EOS backend selection ====================
// If EOS_Traits<problem_t> defines EOSBackend, use it; otherwise fall back to
// DefaultEOSBackend. This preserves backward compatibility with existing full
// specializations that predate the EOSBackend mechanism.

namespace detail
{
template <typename T, typename = void> struct EOSBackendHelper {
	using type = typename DefaultEOSBackend<T>::type;
};

template <typename T> struct EOSBackendHelper<T, std::void_t<typename EOS_Traits<T>::EOSBackend>> {
	using type = typename EOS_Traits<T>::EOSBackend;
};
} // namespace detail

// ==================== EOS (public name) ====================
// Forwards to the trait-selected backend. Methods are declared here (not inherited)
// so that explicit per-problem specializations of individual methods remain valid.

template <typename problem_t> class EOS
{
	using backend_t = typename detail::EOSBackendHelper<problem_t>::type;

      public:
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = backend_t::is_tabulated;
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma; // needed for HLLD solver

	static constexpr amrex::Real boltzmann_constant_ = EOSIdeal<problem_t>::boltzmann_constant_;

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return backend_t::ComputeTgasFromEint(rho, Eint, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return backend_t::ComputeEintFromTgas(rho, Tgas, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return backend_t::ComputeEintFromPres(rho, Pressure, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return backend_t::ComputeEintTempDerivative(rho, Tgas, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(amrex::Real rho, amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		return backend_t::ComputeOtherDerivatives(rho, P, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		return backend_t::ComputePressure(rho, Eint, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return backend_t::ComputeSoundSpeed(rho, Pressure, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		return backend_t::ComputeIsothermalSoundSpeed(rho, Pressure);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return backend_t::ComputeEntropyFromRhoEint(rho, Eint, massScalars);
	}

	// Compute gas internal energy from gas total energy (Eint + Ekin, NOT including B field).
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto ComputeEintFromEgas(double rho, double mx, double my, double mz, double Etot, double magnetic_energy)
	    -> double
	{
		const double Ekin = 0.5 * (mx * mx + my * my + mz * mz) / rho;
		const double Eint = Etot - Ekin - magnetic_energy;
		AMREX_ASSERT_WITH_MESSAGE(Eint > 0., "Gas internal energy is not positive!");
		return Eint;
	}

	// Compute gas total energy (Eint + Ekin, NOT including B field) from gas internal energy.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto ComputeEgasFromEint(double rho, double mx, double my, double mz, double Eint, double magnetic_energy)
	    -> double
	{
		const double Ekin = 0.5 * (mx * mx + my * my + mz * mz) / rho;
		return Eint + Ekin + magnetic_energy;
	}
};

} // namespace quokka

#endif // EOS_HPP_
