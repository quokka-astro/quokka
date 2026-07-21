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
#include <tuple>
#include <type_traits>

#include "util/Optional.hpp"

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "physics_info.hpp"
#include <AMReX_Print.H>

#include "cooling/EOSTabulatedRegistry.hpp"
#include "fundamental_constants.H"
#if defined(CHEMISTRY)
#include "networks/primordial_chem/PrimordialChemNetwork.hpp"
#elif defined(PHOTOCHEMISTRY)
#include "networks/photoionization/PhotoionizationNetwork.hpp"
#endif

namespace quokka
{

// forward declarations of EOS backends
template <typename problem_t> struct EOSIdeal;
template <typename problem_t, typename Network> struct EOSMultigamma;
template <typename problem_t> struct EOSTabulated;

// Single source of truth for the default EOS backend selection, forward-declared here
// so EOS_Traits can reference it. Fully defined after the backend types below.
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

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		if constexpr (gamma_ == 1.0) {
			return NAN;
		}
		return (Eint / rho) * mean_molecular_weight_ * (gamma_ - 1.0) / boltzmann_constant_;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		if constexpr (gamma_ == 1.0) {
			return NAN;
		}
		return rho * boltzmann_constant_ * Tgas / (mean_molecular_weight_ * (gamma_ - 1.0));
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(rho);
		amrex::ignore_unused(massScalars);
		if constexpr (gamma_ == 1.0) {
			return NAN;
		}
		return Pressure / (gamma_ - 1.0);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		amrex::ignore_unused(Tgas);
		amrex::ignore_unused(massScalars);
		if constexpr (gamma_ == 1.0) {
			return NAN;
		}
		return rho * boltzmann_constant_ / (mean_molecular_weight_ * (gamma_ - 1.0));
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> std::tuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real>
	{
		amrex::ignore_unused(massScalars);
		if constexpr (gamma_ == 1.0) {
			return std::make_tuple(NAN, NAN, NAN, NAN, NAN);
		}
		const amrex::Real deint_dRho = 0.0;
		const amrex::Real deint_dP = 1.0 / (rho * (gamma_ - 1.0));
		const amrex::Real dRho_dP = (rho / P) * boltzmann_constant_ / C::k_B;
		const amrex::Real dP_dRho_s = gamma_ * P / rho;
		const amrex::Real G = 0.5 * (gamma_ + 1.0);
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		if constexpr (gamma_ == 1.0) {
			static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
			amrex::ignore_unused(Eint);
			amrex::ignore_unused(massScalars);
			return rho * EOS_Traits<problem_t>::cs_isothermal * EOS_Traits<problem_t>::cs_isothermal;
		}
		amrex::ignore_unused(massScalars);
		if (rho == 0.0) {
			return 0.0;
		}
		return (gamma_ - 1.0) * Eint;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		if constexpr (gamma_ == 1.0) {
			static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
			amrex::ignore_unused(rho);
			amrex::ignore_unused(Pressure);
			amrex::ignore_unused(massScalars);
			return EOS_Traits<problem_t>::cs_isothermal;
		}
		amrex::ignore_unused(massScalars);
		return std::sqrt(gamma_ * Pressure / rho);
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

// ==================== EOSMultigamma backend ====================
// Composition-aware ideal gas whose species metadata is owned by a compiled
// Quokka chemistry-network module.

template <typename problem_t, typename Network> struct EOSMultigamma {
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = false;
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma; // compatibility value used by HLLD
	static constexpr amrex::Real boltzmann_constant_ = EOSIdeal<problem_t>::boltzmann_constant_;
	static_assert(nmscalars_ == Network::species_count, "multigamma EOS requires one mass scalar per network species");

	struct Composition {
		amrex::Real number_density = 0.0;
		amrex::Real mass_density = 0.0;
		amrex::Real heat_capacity = 0.0;
	};

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	composition(quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> Composition
	{
		AMREX_ASSERT_WITH_MESSAGE(static_cast<bool>(massScalars), "multigamma EOS requires species mass densities");
		Composition result{};
		if (massScalars) {
			for (int species = 0; species < nmscalars_; ++species) {
				const amrex::Real numberDensity = (*massScalars)[species] / Network::species_masses[species];
				result.number_density += numberDensity;
				result.mass_density += (*massScalars)[species];
				result.heat_capacity += numberDensity / (Network::species_gammas[species] - 1.0);
			}
		}
		AMREX_ASSERT(result.number_density > 0.0);
		AMREX_ASSERT(result.mass_density > 0.0);
		AMREX_ASSERT(result.heat_capacity > 0.0);
		return result;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto effective_gamma(Composition const &values) -> amrex::Real
	{
		return 1.0 + values.number_density / values.heat_capacity;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		const auto values = composition(massScalars);
		return (Eint / rho) * values.mass_density / (values.heat_capacity * boltzmann_constant_);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		const auto values = composition(massScalars);
		const amrex::Real specificHeat = values.heat_capacity * boltzmann_constant_ / values.mass_density;
		return rho * specificHeat * Tgas;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(rho);
		const auto values = composition(massScalars);
		return Pressure * values.heat_capacity / values.number_density;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		amrex::ignore_unused(Tgas);
		const auto values = composition(massScalars);
		return rho * values.heat_capacity * boltzmann_constant_ / values.mass_density;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		const auto values = composition(massScalars);
		const amrex::Real gamma = effective_gamma(values);
		const amrex::Real deint_dRho = 0.0;
		const amrex::Real deint_dP = values.heat_capacity / (rho * values.number_density);
		const amrex::Real dRho_dP = rho / P;
		const amrex::Real dP_dRho_s = gamma * P / rho;
		const amrex::Real G = 0.5 * (gamma + 1.0);
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		amrex::ignore_unused(rho);
		const auto values = composition(massScalars);
		return Eint * values.number_density / values.heat_capacity;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		const auto values = composition(massScalars);
		return std::sqrt(effective_gamma(values) * Pressure / rho);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		amrex::ignore_unused(rho);
		amrex::ignore_unused(Pressure);
		static_assert(sizeof(problem_t) == 0, "ComputeIsothermalSoundSpeed is not supported by EOSMultigamma");
		return 0.0;
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
	using type = EOSMultigamma<T, chemistry::PrimordialChemNetwork>;
#elif defined(PHOTOCHEMISTRY)
	using type = EOSMultigamma<T, chemistry::PhotoionizationNetwork>;
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
