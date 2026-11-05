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
#include "microphysics/microphysics_traits.hpp"

namespace quokka
{

// The EOSs. Each is a complete struct; quokka::EOS<problem_t> (below) is the one the problem uses.
template <typename problem_t> struct EOSIdeal;
template <typename problem_t> struct EOSMicrophysics;
template <typename problem_t> struct EOSTabulated;

// The EOS a problem gets when its EOS_Traits names none, forward-declared here so EOS_Traits can reference it. Defined
// after EOSTabulated below.
template <typename problem_t> struct DefaultEOS;

// Primary EOS_Traits template: default values for an ideal gamma-law gas, and the problem's EOS. Full specializations
// need only define the values they override; EOS falls back to DefaultEOS if omitted.
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5. / 3.;     // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	static constexpr double mean_molecular_weight = NAN;

	using EOS = typename DefaultEOS<problem_t>::type;
};

// What every EOS has: the conversions between the gas internal and total energy, which only add or subtract the
// kinetic energy.
struct EOSBase {
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

// ==================== EOSIdeal ====================
// gamma-law ideal gas. Always available.

template <typename problem_t> struct EOSIdeal : EOSBase {
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

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real /*rho*/, amrex::Real /*Eint*/,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/ = {}) -> amrex::Real
	{
		// sizeof(problem_t)==0: fires only on instantiation, not at parse time (C++20 ill-formed NDR workaround)
		static_assert(sizeof(problem_t) == 0, "ComputeEntropyFromRhoEint is only supported by the EOSTabulated backend");
		return 0.0;
	}
};

// ==================== EOSMicrophysics ====================
// The default EOS of the problems with chemistry or photochemistry (see DefaultEOS), defined in
// hydro/EOSMicrophysics.hpp: it uses spmasses, which only the chemistry EOSs declare, so only those problems include it.

// ==================== EOSTabulated ====================
// Temperature methods read the resampled table; all other methods delegate to EOSIdeal.

template <typename problem_t> struct EOSTabulated : EOSBase {
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

// DefaultEOS — definition (forward-declared near EOS_Traits above). Both EOS_Traits (primary template) and EOSChoice
// (the fallback for full specializations without EOS) use it, so the default lives in one place.
template <typename T> struct DefaultEOS {
	using type = std::conditional_t<Microphysics_Traits<T>::chemistry || Microphysics_Traits<T>::photochemistry, EOSMicrophysics<T>, EOSIdeal<T>>;
};

namespace detail
{
// EOS_Traits<T>::EOS if the problem's EOS_Traits defines it, else DefaultEOS: a full specialization of EOS_Traits does
// not inherit the primary template's EOS.
template <typename T, typename = void> struct EOSChoice {
	using type = typename DefaultEOS<T>::type;
};

template <typename T> struct EOSChoice<T, std::void_t<typename EOS_Traits<T>::EOS>> {
	using type = typename EOS_Traits<T>::EOS;
};
} // namespace detail

// ==================== EOS (public name) ====================
// The problem's EOS. A problem that needs its own EOS writes a struct, usually derived from one of the above, and names
// it in EOS_Traits<problem_t>::EOS.

template <typename problem_t> using EOS = typename detail::EOSChoice<problem_t>::type;

} // namespace quokka

#endif // EOS_HPP_
