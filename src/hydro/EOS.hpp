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
#include "eos.H"

#ifdef CHEMISTRY
#include "actual_eos_data.H"
#endif

namespace quokka
{

// forward declarations of EOS backends
template <typename problem_t> struct EOSIdeal;
#if defined(CHEMISTRY) || defined(PHOTOCHEMISTRY)
template <typename problem_t> struct EOSMicrophysics;
#endif
template <typename problem_t> struct EOSTabulated;

// Primary EOS_Traits template. Provides default values for ideal gamma-law EOS
// and selects the compile-time EOS backend. Full specializations need only define
// the trait values they override; EOSBackend defaults to EOSIdeal via SFINAE if omitted.
//
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5. / 3.;     // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	static constexpr double mean_molecular_weight = NAN;
	static constexpr double boltzmann_constant = C::k_B;

#if defined(CHEMISTRY) || defined(PHOTOCHEMISTRY)
	using EOSBackend = EOSMicrophysics<problem_t>;
#else
	using EOSBackend = EOSIdeal<problem_t>;
#endif
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

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real Tgas = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			chem_eos_t estate;
			estate.rho = rho;
			estate.e = Eint / rho;
			estate.mu = mean_molecular_weight_ / C::m_u;
			eos(eos_input_re, estate);
			Tgas = estate.T * C::k_B / boltzmann_constant_;
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
			chem_eos_t estate;
			estate.rho = rho;
			estate.T = Tgas;
			estate.mu = mean_molecular_weight_ / C::m_u;
			eos(eos_input_rt, estate);
			Eint = estate.e * rho * boltzmann_constant_ / C::k_B;
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
			chem_eos_t estate;
			estate.rho = rho;
			estate.p = Pressure;
			estate.mu = mean_molecular_weight_ / C::m_u;
			eos(eos_input_rp, estate);
			Eint = estate.e * rho;
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
			chem_eos_t estate;
			estate.rho = rho;
			estate.T = Tgas;
			estate.mu = mean_molecular_weight_ / C::m_u;
			eos(eos_input_rt, estate);
			dEint_dT = estate.dedT * rho * boltzmann_constant_ / C::k_B;
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
			chem_eos_t estate;
			estate.rho = rho;
			estate.p = P;
			estate.mu = mean_molecular_weight_ / C::m_u;
			eos(eos_input_rp, estate);
			deint_dRho = estate.dedr;
			deint_dP = 1.0 / estate.dpde;
			dRho_dP = 1.0 / (estate.dpdr * C::k_B / boltzmann_constant_);
			dP_dRho_s = estate.cs * estate.cs;
			G = estate.G;
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

		chem_eos_t estate;
		estate.rho = rho;
		if (rho == 0.0) {
			estate.e = 0;
		} else {
			estate.e = Eint / rho;
		}
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_re, estate);
		P = estate.p;
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

		chem_eos_t estate;
		estate.rho = rho;
		estate.p = Pressure;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_rp, estate);
		cs = estate.cs;
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
};

// ==================== EOSMicrophysics backend ====================
// Only compiled when CHEMISTRY or PHOTOCHEMISTRY is defined.

#if defined(CHEMISTRY) || defined(PHOTOCHEMISTRY)
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
		for (int ii = 0; ii < NumSpec; ++ii) {
			chemstate.xn[ii] = -1.0;
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
		for (int ii = 0; ii < NumSpec; ++ii) {
			chemstate.xn[ii] = -1.0;
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
		for (int ii = 0; ii < NumSpec; ++ii) {
			chemstate.xn[ii] = -1.0;
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
		amrex::ignore_unused(Tgas);
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.T = NAN;
		for (int ii = 0; ii < NumSpec; ++ii) {
			chemstate.xn[ii] = -1.0;
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
		for (int ii = 0; ii < NumSpec; ++ii) {
			chemstate.xn[ii] = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				chemstate.xn[nn] = massArray[nn] / spmasses[nn];
			}
		}

		eos(eos_input_rp, chemstate);
		amrex::Real deint_dRho = chemstate.dedr;
		amrex::Real deint_dP = 1.0 / chemstate.dpde;
		amrex::Real dRho_dP = 1.0 / (chemstate.dpdr * C::k_B / boltzmann_constant_);
		amrex::Real dP_dRho_s = chemstate.cs * chemstate.cs;
		amrex::Real G = chemstate.G;
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.e = Eint / rho;
		for (int ii = 0; ii < NumSpec; ++ii) {
			chemstate.xn[ii] = -1.0;
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
		eos_t chemstate;
		chemstate.rho = rho;
		chemstate.p = Pressure;
		for (int ii = 0; ii < NumSpec; ++ii) {
			chemstate.xn[ii] = -1.0;
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
};
#endif // CHEMISTRY || PHOTOCHEMISTRY

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
		auto *reg = ResampledCooling::getEOSTabulatedRegistry();
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(reg != nullptr, "EOSTabulated: registry not registered before use!");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(reg->active, "EOSTabulated: registry not active!");

		AMREX_IF_ON_DEVICE((return ResampledCooling::ComputeTgasFromEgas(rho, Eint, reg->device);))
		AMREX_IF_ON_HOST((return ResampledCooling::ComputeTgasFromEgas(rho, Eint, reg->host);))
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		auto *reg = ResampledCooling::getEOSTabulatedRegistry();
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(reg != nullptr, "EOSTabulated: registry not registered before use!");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(reg->active, "EOSTabulated: registry not active!");

		AMREX_IF_ON_DEVICE((return ResampledCooling::ComputeEgasFromTgas(rho, Tgas, reg->device);))
		AMREX_IF_ON_HOST((return ResampledCooling::ComputeEgasFromTgas(rho, Tgas, reg->host);))
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		// Compute dEint/dT with one root-find + two cheap table lookups
		// instead of two root-finds (toms748, up to 32 iterations each).
		const amrex::Real Eint = ComputeEintFromTgas(rho, Tgas, massScalars);
		constexpr amrex::Real eps = 1.0e-6;
		const amrex::Real dEint = amrex::max(eps * Eint, eps);
		const amrex::Real T_plus = ComputeTgasFromEint(rho, Eint + dEint, massScalars);
		const amrex::Real T_minus = ComputeTgasFromEint(rho, Eint - dEint, massScalars);
		const amrex::Real dT = T_plus - T_minus;
		return (dT > 0.0) ? (2.0 * dEint) / dT : 0.0;
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
};

// ==================== EOS backend selection ====================
// If EOS_Traits<problem_t> defines EOSBackend, use it; otherwise default to EOSIdeal.
// This preserves backward compatibility with existing full specializations that
// predate the EOSBackend mechanism.

namespace detail
{
template <typename T, typename = void> struct EOSBackendHelper {
#if defined(CHEMISTRY) || defined(PHOTOCHEMISTRY)
	using type = EOSMicrophysics<T>;
#else
	using type = EOSIdeal<T>;
#endif
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
